#include "arcanum/passes/Passes.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/Verifier.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"

#include "llvm/ADT/StringSet.h"

#include <string>

namespace arcanum {
namespace {

// ---------------------------------------------------------------------------
// Assign target collection: walks one or more regions for arc.assign ops
// whose target is defined by an arc.var. Deduplicates across all regions.
// ---------------------------------------------------------------------------

void collectAssignTargetsFromRegion(mlir::Region& region,
                                    llvm::StringSet<>& seen,
                                    llvm::SmallVector<std::string>& names) {
  region.walk([&](arc::AssignOp assignOp) {
    if (auto* defOp = assignOp.getTarget().getDefiningOp()) {
      if (auto varOp = llvm::dyn_cast<arc::VarOp>(defOp)) {
        auto name = varOp.getName().str();
        if (seen.insert(name).second) {
          names.push_back(name);
        }
      }
    }
  });
}

std::string joinNames(const llvm::SmallVector<std::string>& names) {
  std::string result;
  for (size_t i = 0; i < names.size(); ++i) {
    if (i > 0)
      result += ", ";
    result += names[i];
  }
  return result;
}

// ---------------------------------------------------------------------------
// Auto-compute assigns: if the assigns attribute is absent, collect all
// arc.assign targets from body and update regions (deduplicated) and set
// the attribute.
// ---------------------------------------------------------------------------

void autoComputeAssigns(arc::LoopOp loopOp) {
  if (loopOp->getAttrOfType<mlir::StringAttr>("assigns")) {
    return;
  }

  llvm::StringSet<> seen;
  llvm::SmallVector<std::string> names;
  collectAssignTargetsFromRegion(loopOp.getBodyRegion(), seen, names);
  collectAssignTargetsFromRegion(loopOp.getUpdateRegion(), seen, names);

  auto assigns = joinNames(names);
  if (!assigns.empty()) {
    loopOp->setAttr("assigns",
                    mlir::StringAttr::get(loopOp->getContext(), assigns));
  }
}

// ---------------------------------------------------------------------------
// Contract validation: warn if invariant is missing, error if while/do-while
// lacks a variant annotation.
// ---------------------------------------------------------------------------

bool validateLoopContracts(arc::LoopOp loopOp) {
  bool hasError = false;

  if (!loopOp->getAttrOfType<mlir::StringAttr>("invariant")) {
    loopOp.emitWarning("loop is missing loop_invariant annotation");
  }

  if (!loopOp->getAttrOfType<mlir::StringAttr>("variant")) {
    bool isForLoop = !loopOp.getInitRegion().empty();
    if (!isForLoop) {
      loopOp.emitError(
          "while/do-while loop requires explicit loop_variant annotation");
      hasError = true;
    }
  }

  return hasError;
}

// ---------------------------------------------------------------------------
// LoopContractPass: an MLIR pass that walks all arc.loop ops and
// auto-computes assigns and validates contracts.
// ---------------------------------------------------------------------------

struct LoopContractPass
    : public mlir::PassWrapper<LoopContractPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LoopContractPass)

  llvm::StringRef getArgument() const override { return "loop-contract"; }

  llvm::StringRef getDescription() const override {
    return "Auto-compute assigns, validate contracts";
  }

  void runOnOperation() override {
    // TODO: Auto-infer variant for counted for-loops. Pattern-match:
    // init sets var to lo, cond compares var < hi, update increments by 1
    // -> variant = hi - var. Deferred to a follow-up commit.
    bool hasError = false;
    getOperation().walk([&hasError](arc::LoopOp loopOp) {
      autoComputeAssigns(loopOp);
      if (validateLoopContracts(loopOp)) {
        hasError = true;
      }
    });
    if (hasError) {
      signalPassFailure();
    }
  }
};

} // namespace

mlir::LogicalResult runPasses(mlir::ModuleOp module) {
  mlir::PassManager pm(module->getContext());
  pm.addPass(std::make_unique<LoopContractPass>());

  if (pm.run(module).failed()) {
    return mlir::failure();
  }

  return mlir::success();
}

} // namespace arcanum
