#include "frontend/ContractParser.h"
#include "frontend/SubsetEnforcer.h"
#include "dialect/Lowering.h"
#include "passes/Passes.h"
#include "backend/WhyMLEmitter.h"
#include "backend/Why3Runner.h"
#include "report/ReportGenerator.h"

#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/MLIRContext.h"

#include <string>

using namespace llvm;
using namespace clang::tooling;

static cl::OptionCategory arcanumCategory("Arcanum options");

static cl::opt<std::string> mode(
    "mode",
    cl::desc("Operating mode (verify)"),
    cl::init("verify"),
    cl::cat(arcanumCategory));

static cl::opt<std::string> why3Path(
    "why3-path",
    cl::desc("Path to why3 binary"),
    cl::init("why3"),
    cl::cat(arcanumCategory));

static cl::opt<int> timeout(
    "timeout",
    cl::desc("Per-obligation timeout in seconds"),
    cl::init(30),
    cl::cat(arcanumCategory));

int main(int argc, const char** argv) {
  auto expectedParser =
      CommonOptionsParser::create(argc, argv, arcanumCategory);
  if (!expectedParser) {
    llvm::errs() << expectedParser.takeError();
    return 5;
  }
  CommonOptionsParser& optionsParser = expectedParser.get();

  if (mode != "verify") {
    llvm::errs() << "error: unsupported mode '" << mode
                 << "' (only 'verify' is supported in Slice 1)\n";
    return 5;
  }

  const auto& sourceFiles = optionsParser.getSourcePathList();
  if (sourceFiles.empty()) {
    llvm::errs() << "error: no input files\n";
    return 5;
  }

  // Validate input files exist
  for (const auto& file : sourceFiles) {
    if (!llvm::sys::fs::exists(file)) {
      llvm::errs() << "error: file not found: " << file << "\n";
      return 5;
    }
  }

  // Stage 1: Clang Frontend — parse source into AST
  ClangTool tool(optionsParser.getCompilations(),
                 optionsParser.getSourcePathList());
  tool.appendArgumentsAdjuster(getInsertArgumentAdjuster(
      "-fparse-all-comments", ArgumentInsertPosition::BEGIN));

  // Stages 2-8 will be wired here
  // For now, the AST is captured in the FrontendAction and passed forward.

  arcanum::ArcanumFrontendAction action;
  auto result = tool.run(newFrontendActionFactory(&action).get());
  if (result != 0) {
    return 4; // Parse error
  }

  // Stage 2: Subset Enforcer
  auto enforceResult = arcanum::enforceSubset(action.getASTContext());
  if (!enforceResult.passed) {
    for (const auto& diag : enforceResult.diagnostics) {
      llvm::errs() << diag << "\n";
    }
    return 3;
  }

  // Stage 3: Contract Parser
  auto contracts = arcanum::parseContracts(action.getASTContext());

  // Stage 4: Arc MLIR Lowering
  mlir::MLIRContext mlirContext;
  auto arcModule =
      arcanum::lowerToArc(mlirContext, action.getASTContext(), contracts);
  if (!arcModule) {
    llvm::errs() << "error: lowering to Arc MLIR failed\n";
    return 5;
  }

  // Stage 5: MLIR Pass Manager
  if (arcanum::runPasses(*arcModule).failed()) {
    llvm::errs() << "error: MLIR verification failed\n";
    return 5;
  }

  // Stage 6: WhyML Emitter
  auto whymlResult = arcanum::emitWhyML(*arcModule);
  if (!whymlResult) {
    llvm::errs() << "error: WhyML emission failed\n";
    return 5;
  }

  // Stage 7: Why3 Runner
  auto obligations =
      arcanum::runWhy3(whymlResult->filePath, why3Path, timeout);

  // Stage 8: Report Generator
  auto report =
      arcanum::generateReport(obligations, whymlResult->locationMap);
  llvm::outs() << report.text << "\n";

  return report.allPassed ? 0 : 1;
}
