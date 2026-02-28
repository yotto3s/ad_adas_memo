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

/// NOTE (CQ-3): Duplicates inline join logic in Lowering.cpp
/// attachLoopContractAttrs.  Consolidation deferred pending a shared utility.
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
// Auto-infer variant for counted for-loops.
// Pattern-match: init sets var to lo, cond compares var < hi, update
// increments var by 1 -> variant = "hi - var".
// ---------------------------------------------------------------------------

/// Extract the loop induction variable name from the init region.
/// Looks for the first arc.var op and returns its name.
std::string extractInitVarName(mlir::Region& initRegion) {
  if (initRegion.empty()) {
    return "";
  }
  for (auto& op : initRegion.front().getOperations()) {
    if (auto varOp = llvm::dyn_cast<arc::VarOp>(&op)) {
      return varOp.getName().str();
    }
  }
  return "";
}

/// Extract the upper bound variable name from the cond region.
/// Looks for arc.cmp lt, <var>, <hi> and returns the name of <hi>.
/// Returns empty string if pattern not recognized.
std::string extractCondUpperBound(mlir::Region& condRegion,
                                  const std::string& inductionVar,
                                  llvm::DenseMap<mlir::Value, std::string>&
                                      valueNames) {
  if (condRegion.empty()) {
    return "";
  }
  // First pass: collect variable names from VarOp results
  for (auto& op : condRegion.front().getOperations()) {
    if (auto varOp = llvm::dyn_cast<arc::VarOp>(&op)) {
      valueNames[varOp.getResult()] = varOp.getName().str();
    }
  }
  // Second pass: find CmpOp with lt predicate
  for (auto& op : condRegion.front().getOperations()) {
    if (auto cmpOp = llvm::dyn_cast<arc::CmpOp>(&op)) {
      auto pred = cmpOp.getPredicate();
      if (pred == "lt") {
        // Check if LHS is the induction variable
        std::string lhsName;
        if (auto it = valueNames.find(cmpOp.getLhs()); it != valueNames.end()) {
          lhsName = it->second;
        }
        if (lhsName == inductionVar) {
          // RHS is the upper bound - get its name
          if (auto it = valueNames.find(cmpOp.getRhs());
              it != valueNames.end()) {
            return it->second;
          }
        }
      } else if (pred == "le") {
        // var <= hi  pattern: variant = hi - var + 1 (more complex, skip)
      }
    }
  }
  return "";
}

/// Check if the update region increments the induction variable by 1.
/// Looks for arc.add <var>, <const 1> followed by arc.assign.
bool updateIncrementsBy1(mlir::Region& updateRegion,
                         const std::string& inductionVar) {
  if (updateRegion.empty()) {
    return false;
  }
  for (auto& op : updateRegion.front().getOperations()) {
    if (auto assignOp = llvm::dyn_cast<arc::AssignOp>(&op)) {
      // Check target is the induction variable
      if (auto* defOp = assignOp.getTarget().getDefiningOp()) {
        if (auto varOp = llvm::dyn_cast<arc::VarOp>(defOp)) {
          if (varOp.getName() != inductionVar) {
            continue;
          }
        } else {
          continue;
        }
      } else {
        continue;
      }
      // Check value is an AddOp with constant 1
      if (auto* valueOp = assignOp.getValue().getDefiningOp()) {
        if (auto addOp = llvm::dyn_cast<arc::AddOp>(valueOp)) {
          // Check if one operand is constant 1
          auto checkConst1 = [](mlir::Value val) -> bool {
            if (auto* op = val.getDefiningOp()) {
              if (auto constOp = llvm::dyn_cast<arc::ConstantOp>(op)) {
                if (auto intAttr =
                        llvm::dyn_cast<mlir::IntegerAttr>(constOp.getValue())) {
                  return intAttr.getValue().getSExtValue() == 1;
                }
              }
            }
            return false;
          };
          if (checkConst1(addOp.getRhs()) || checkConst1(addOp.getLhs())) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

/// Try to auto-infer variant for a counted for-loop.
/// Returns the inferred variant string, or empty if pattern not matched.
std::string tryInferVariant(arc::LoopOp loopOp) {
  auto inductionVar = extractInitVarName(loopOp.getInitRegion());
  if (inductionVar.empty()) {
    return "";
  }

  // Build a value-to-name map from all regions for name resolution.
  // The cond region may reference values from the init region or
  // function arguments -- we need to map them to names.
  llvm::DenseMap<mlir::Value, std::string> valueNames;

  // Map init region VarOp results
  if (!loopOp.getInitRegion().empty()) {
    for (auto& op : loopOp.getInitRegion().front().getOperations()) {
      if (auto varOp = llvm::dyn_cast<arc::VarOp>(&op)) {
        valueNames[varOp.getResult()] = varOp.getName().str();
      }
    }
  }

  // Map parent region variables (function parameters via VarOp or block args)
  auto* parentBlock = loopOp->getBlock();
  if (parentBlock) {
    for (auto& op : *parentBlock) {
      if (auto varOp = llvm::dyn_cast<arc::VarOp>(&op)) {
        valueNames[varOp.getResult()] = varOp.getName().str();
      }
    }
  }

  auto upperBound =
      extractCondUpperBound(loopOp.getCondRegion(), inductionVar, valueNames);
  if (upperBound.empty()) {
    return "";
  }

  if (!updateIncrementsBy1(loopOp.getUpdateRegion(), inductionVar)) {
    return "";
  }

  return upperBound + " - " + inductionVar;
}

/// Auto-infer the variant attribute for a for-loop if not already provided.
void autoInferVariant(arc::LoopOp loopOp) {
  if (loopOp->getAttrOfType<mlir::StringAttr>("variant")) {
    return;
  }
  bool isForLoop = !loopOp.getInitRegion().empty();
  if (!isForLoop) {
    return;
  }

  auto inferred = tryInferVariant(loopOp);
  if (!inferred.empty()) {
    loopOp->setAttr("variant",
                     mlir::StringAttr::get(loopOp->getContext(), inferred));
  } else {
    loopOp.emitWarning("could not auto-infer loop_variant for counted "
                        "for-loop; provide an explicit loop_variant");
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
    bool hasError = false;
    getOperation().walk([&hasError](arc::LoopOp loopOp) {
      autoComputeAssigns(loopOp);
      autoInferVariant(loopOp);
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
