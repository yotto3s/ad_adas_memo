#include "DiagnosticTracker.h"
#include "ExitCodes.h"
#include "backend/Why3Runner.h"
#include "backend/WhyMLEmitter.h"
#include "dialect/Lowering.h"
#include "frontend/ContractParser.h"
#include "frontend/SubsetEnforcer.h"
#include "passes/Passes.h"
#include "report/ReportGenerator.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"

#include <string>
#include <vector>

using namespace llvm;
using namespace clang::tooling;

static cl::OptionCategory arcanumCategory("Arcanum options");

static cl::opt<std::string> mode("mode", cl::desc("Operating mode (verify)"),
                                 cl::init("verify"), cl::cat(arcanumCategory));

static cl::opt<std::string> why3Path("why3-path",
                                     cl::desc("Path to why3 binary"),
                                     cl::init("why3"),
                                     cl::cat(arcanumCategory));

static cl::opt<int> timeout("timeout",
                            cl::desc("Per-obligation timeout in seconds"),
                            cl::init(arcanum::DEFAULT_TIMEOUT_SECONDS),
                            cl::cat(arcanumCategory));

int main(int argc, const char** argv) {
  auto expectedParser =
      CommonOptionsParser::create(argc, argv, arcanumCategory);
  if (!expectedParser) {
    llvm::errs() << expectedParser.takeError();
    return arcanum::ExitInternalError;
  }
  CommonOptionsParser& optionsParser = expectedParser.get();

  if (mode != "verify") {
    llvm::errs() << "error: unsupported mode '" << mode
                 << "' (only 'verify' is supported in Slice 1)\n";
    return arcanum::ExitInternalError;
  }

  const auto& sourceFiles = optionsParser.getSourcePathList();
  if (sourceFiles.empty()) {
    llvm::errs() << "error: no input files\n";
    return arcanum::ExitInternalError;
  }

  // Validate input files exist
  for (const auto& file : sourceFiles) {
    if (!llvm::sys::fs::exists(file)) {
      llvm::errs() << "error: file not found: " << file << "\n";
      return arcanum::ExitInternalError;
    }
  }

  // Stage 1: Clang Frontend — parse source into AST using buildASTs
  ClangTool tool(optionsParser.getCompilations(),
                 optionsParser.getSourcePathList());
  tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-fparse-all-comments", ArgumentInsertPosition::BEGIN));
  // Suppress errors from GCC-specific warning flags in compile_commands.json
  // (e.g., -Wno-class-memaccess) when Clang processes them.
  tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-Wno-unknown-warning-option", ArgumentInsertPosition::BEGIN));
  tool.appendArgumentsAdjuster(getClangStripOutputAdjuster());

  std::vector<std::unique_ptr<clang::ASTUnit>> astUnits;
  int buildResult = tool.buildASTs(astUnits);
  if (buildResult != 0 || astUnits.empty() || !astUnits[0]) {
    llvm::errs() << "error: failed to parse input file\n";
    return arcanum::ExitParseError;
  }

  auto& astContext = astUnits[0]->getASTContext();

  // Stage 2: Subset Enforcer
  auto enforceResult = arcanum::enforceSubset(astContext);
  if (!enforceResult.passed) {
    for (const auto& diag : enforceResult.diagnostics) {
      llvm::errs() << diag << "\n";
    }
    return arcanum::ExitSubsetViolation;
  }

  // Stage 3: Contract Parser
  auto contracts = arcanum::parseContracts(astContext);

  // Stage 4: Arc MLIR Lowering
  arcanum::DiagnosticTracker::reset();
  mlir::MLIRContext mlirContext;
  auto arcModule = arcanum::lowerToArc(mlirContext, astContext, contracts);
  if (!arcModule) {
    llvm::errs() << "error: lowering to Arc MLIR failed\n";
    return arcanum::ExitInternalError;
  }

  // Stage 5: MLIR Pass Manager
  if (arcanum::runPasses(*arcModule).failed()) {
    llvm::errs() << "error: MLIR verification failed\n";
    return arcanum::ExitInternalError;
  }

  // Stage 6: WhyML Emitter
  auto whymlResult = arcanum::emitWhyML(*arcModule);
  if (!whymlResult) {
    llvm::errs() << "error: WhyML emission failed\n";
    return arcanum::ExitInternalError;
  }

  // Stage 7: Why3 Runner
  auto obligations = arcanum::runWhy3(whymlResult->filePath, why3Path, timeout);

  // Clean up the temporary .mlw file created by the WhyML emitter.
  auto removeEc = llvm::sys::fs::remove(whymlResult->filePath);
  (void)removeEc;

  // Stage 8: Report Generator
  auto report = arcanum::generateReport(obligations, whymlResult->locationMap);
  llvm::outs() << report.text;
  if (arcanum::DiagnosticTracker::getFallbackCount() > 0) {
    llvm::outs() << "\nWarning: "
                 << arcanum::DiagnosticTracker::getFallbackCount()
                 << " expression(s) used zero-constant fallback during "
                    "lowering. Results may be unreliable.\n";
  }
  llvm::outs() << "\n";

  return report.allPassed ? arcanum::ExitSuccess
                          : arcanum::ExitVerificationFailed;
}
