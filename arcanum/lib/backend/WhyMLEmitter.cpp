#include "arcanum/backend/WhyMLEmitter.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace arcanum {
namespace {

/// Token string for \result in contract expressions.
constexpr llvm::StringLiteral RESULT_TOKEN("\\result");

/// String representations of INT32 bounds for overflow assertions.
constexpr llvm::StringLiteral INT32_MIN_STR("-2147483648");
constexpr llvm::StringLiteral INT32_MAX_STR("2147483647");

/// Convert a contract expression string from Arc format to WhyML format.
/// Transforms: \result -> result, && -> /\, || -> \/, etc.
///
/// The input is a serialized contract expression from serializeExpr() in
/// Lowering.cpp.  It uses C-like operators (&&, ||, /, %) and \result.
/// Unary negation is serialized as "-<operand>" by serializeExpr(); we
/// handle it here but note that WhyML uses prefix "-" natively, so no
/// translation is needed for the "-" character itself.
std::string contractToWhyML(llvm::StringRef contract) {
  std::string result;
  size_t i = 0;
  while (i < contract.size()) {
    if (contract.substr(i).starts_with(RESULT_TOKEN)) {
      result += "result";
      i += RESULT_TOKEN.size();
    } else if (contract.substr(i).starts_with("&&")) {
      result += "/\\";
      i += 2;
    } else if (contract.substr(i).starts_with("||")) {
      result += "\\/";
      i += 2;
    } else if (contract.substr(i).starts_with("==")) {
      result += "=";
      i += 2;
    } else if (contract.substr(i).starts_with("!=")) {
      result += "<>";
      i += 2;
    } else if (contract[i] == '!' &&
               (i + 1 >= contract.size() || contract[i + 1] != '=')) {
      result += "not ";
      ++i;
    } else if (contract[i] == '%') {
      result += " mod ";
      ++i;
    } else if (contract[i] == '/') {
      // The input comes from serializeExpr() which uses C-like operators.
      // The `/\` sequence cannot appear in the input (that is a WhyML AND
      // operator, not a serialized Arc construct).  We must check for `\`
      // after `/` anyway to be defensive: if it appears, the `/` is not a
      // division operator but an unexpected token, so we skip translation.
      if (i + 1 < contract.size() && contract[i + 1] == '\\') {
        // Defensive: unexpected `/\` in input -- output literally.
        llvm::errs() << "warning: unexpected '/\\' sequence in contract "
                        "expression: '"
                     << contract << "'\n";
        result += contract[i];
        ++i;
      } else {
        result += " div ";
        ++i;
      }
    } else {
      result += contract[i];
      ++i;
    }
  }
  return result;
}

/// Convert a CamelCase or snake_case function name to a WhyML module name
/// (first letter capitalized).
std::string toModuleName(llvm::StringRef funcName) {
  std::string result = funcName.str();
  if (!result.empty()) {
    // Convert snake_case to CamelCase for module name
    std::string camel;
    bool nextUpper = true;
    for (char c : result) {
      if (c == '_') {
        nextUpper = true;
      } else if (nextUpper) {
        camel += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        nextUpper = false;
      } else {
        camel += c;
      }
    }
    return camel;
  }
  return "Module";
}

/// Extract the param_names attribute from a FuncOp into a vector of strings.
/// Falls back to "argN" naming if the attribute is missing or has fewer
/// entries than the function has arguments.
llvm::SmallVector<std::string> getParamNames(arc::FuncOp funcOp) {
  llvm::SmallVector<std::string> names;
  if (auto paramNamesAttr =
          funcOp->getAttrOfType<mlir::ArrayAttr>("param_names")) {
    for (auto attr : paramNamesAttr) {
      names.push_back(llvm::cast<mlir::StringAttr>(attr).getValue().str());
    }
  }
  return names;
}

class WhyMLWriter {
public:
  explicit WhyMLWriter(mlir::ModuleOp module) : module(module) {}

  std::optional<WhyMLResult> emit() {
    WhyMLResult result;

    // The walk callback captures `result` by reference.  This is safe
    // because walk() executes synchronously and result outlives the call.
    module.walk([&](arc::FuncOp funcOp) { emitFunction(funcOp, result); });

    if (result.whymlText.empty()) {
      return std::nullopt;
    }

    // Write to temp file
    llvm::SmallString<128> tmpPath; // NOLINT(readability-magic-numbers)
    std::error_code ec;
    ec = llvm::sys::fs::createTemporaryFile("arcanum", "mlw", tmpPath);
    if (ec) {
      return std::nullopt;
    }

    llvm::raw_fd_ostream out(tmpPath, ec);
    if (ec) {
      return std::nullopt;
    }
    out << result.whymlText;
    out.close();
    if (out.has_error()) {
      out.clear_error();
      return std::nullopt;
    }

    result.filePath = tmpPath.str().str();
    return result;
  }

private:
  std::string toWhyMLType(mlir::Type t) {
    if (mlir::isa<arc::BoolType>(t)) {
      return "bool";
    }
    return "int";
  }

  void emitFunction(arc::FuncOp funcOp, WhyMLResult& result) {
    std::string outBuf;
    llvm::raw_string_ostream out(outBuf);
    auto funcName = funcOp.getSymName().str();
    auto moduleName = toModuleName(funcName);

    // Scan types and ops to decide which Why3 modules are needed
    auto funcType = funcOp.getFunctionType();
    bool needsBool = false;
    bool needsComputerDivision = false;
    auto& entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      if (mlir::isa<arc::BoolType>(entryBlock.getArgument(i).getType())) {
        needsBool = true;
      }
    }
    if (funcType.getNumResults() > 0 &&
        mlir::isa<arc::BoolType>(funcType.getResult(0))) {
      needsBool = true;
    }
    // Scan for DivOp/RemOp to determine if ComputerDivision is needed
    funcOp.walk([&](mlir::Operation* op) {
      if (llvm::isa<arc::DivOp>(op) || llvm::isa<arc::RemOp>(op)) {
        needsComputerDivision = true;
      }
    });

    out << "module " << moduleName << "\n";
    out << "  use int.Int\n";
    if (needsBool) {
      out << "  use bool.Bool\n";
    }
    if (needsComputerDivision) {
      out << "  use int.ComputerDivision\n";
    }
    out << "\n";

    // Function signature
    out << "  let " << funcName << " ";

    // Parameters - use original C++ names from param_names attribute
    auto paramNames = getParamNames(funcOp);
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      std::string pname =
          (i < paramNames.size()) ? paramNames[i] : ("arg" + std::to_string(i));
      out << "(" << pname << ": "
          << toWhyMLType(entryBlock.getArgument(i).getType()) << ") ";
    }

    // Result type
    std::string resultType = "int";
    if (funcType.getNumResults() > 0) {
      resultType = toWhyMLType(funcType.getResult(0));
    }
    out << ": " << resultType << "\n";

    // Requires clauses
    if (auto reqAttr = funcOp.getRequiresAttrAttr()) {
      auto reqStr = reqAttr.getValue();
      // Split on && at top level for separate requires clauses
      out << "    requires { " << contractToWhyML(reqStr) << " }\n";
    }

    // Ensures clauses
    if (auto ensAttr = funcOp.getEnsuresAttrAttr()) {
      auto ensStr = ensAttr.getValue();
      out << "    ensures  { " << contractToWhyML(ensStr) << " }\n";
    }

    // Function body - walk the ops and emit WhyML
    out << "  =\n";
    emitBody(funcOp, out);

    out << "\nend\n\n";

    result.whymlText += outBuf;

    // Populate location map
    auto loc = funcOp.getLoc();
    LocationEntry entry;
    entry.functionName = funcName;
    if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
      entry.fileName = fileLoc.getFilename().str();
      entry.line = fileLoc.getLine();
    }
    result.locationMap[funcName] = entry;
    result.moduleToFuncMap[moduleName] = funcName;
  }

  void emitBody(arc::FuncOp funcOp, llvm::raw_string_ostream& out) {
    auto& entryBlock = funcOp.getBody().front();
    // Map MLIR values to WhyML variable names
    llvm::DenseMap<mlir::Value, std::string> nameMap;

    // Map block arguments to original C++ parameter names
    auto paramNames = getParamNames(funcOp);
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      std::string pname =
          (i < paramNames.size()) ? paramNames[i] : ("arg" + std::to_string(i));
      nameMap[entryBlock.getArgument(i)] = pname;
    }

    for (auto& op : entryBlock.getOperations()) {
      emitOp(op, out, nameMap);
    }
  }

  /// Emit a binary arithmetic op with an i32 overflow assertion.
  void emitArithWithOverflowCheck(
      mlir::Value result, mlir::Value lhsVal, mlir::Value rhsVal,
      const std::string& whymlOp, llvm::raw_string_ostream& out,
      llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto lhs = getExpr(lhsVal, nameMap);
    auto rhs = getExpr(rhsVal, nameMap);
    auto expr = "(" + lhs + " " + whymlOp + " " + rhs + ")";
    out << "    assert { " << INT32_MIN_STR.data() << " <= " << expr << " /\\ "
        << expr << " <= " << INT32_MAX_STR.data() << " };\n";
    nameMap[result] = expr;
  }

  /// Emit a division-like op with divisor-not-zero and overflow assertions.
  void emitDivLikeOp(mlir::Value result, mlir::Value lhsVal, mlir::Value rhsVal,
                     const std::string& whymlFunc,
                     llvm::raw_string_ostream& out,
                     llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto lhs = getExpr(lhsVal, nameMap);
    auto rhs = getExpr(rhsVal, nameMap);
    auto expr = "(" + whymlFunc + " " + lhs + " " + rhs + ")";
    out << "    assert { " << rhs << " <> 0 };\n";
    // Overflow check: INT_MIN / -1 overflows in C (undefined behavior)
    out << "    assert { " << INT32_MIN_STR.data() << " <= " << expr << " /\\ "
        << expr << " <= " << INT32_MAX_STR.data() << " };\n";
    nameMap[result] = expr;
  }

  void emitOp(mlir::Operation& op, llvm::raw_string_ostream& out,
              llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    if (auto constOp = llvm::dyn_cast<arc::ConstantOp>(&op)) {
      auto attr = constOp.getValue();
      std::string valStr;
      if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
        valStr = std::to_string(intAttr.getInt());
      } else if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
        valStr = boolAttr.getValue() ? "true" : "false";
      }
      nameMap[constOp.getResult()] = valStr;
    } else if (auto addOp = llvm::dyn_cast<arc::AddOp>(&op)) {
      emitArithWithOverflowCheck(addOp.getResult(), addOp.getLhs(),
                                 addOp.getRhs(), "+", out, nameMap);
    } else if (auto subOp = llvm::dyn_cast<arc::SubOp>(&op)) {
      emitArithWithOverflowCheck(subOp.getResult(), subOp.getLhs(),
                                 subOp.getRhs(), "-", out, nameMap);
    } else if (auto mulOp = llvm::dyn_cast<arc::MulOp>(&op)) {
      emitArithWithOverflowCheck(mulOp.getResult(), mulOp.getLhs(),
                                 mulOp.getRhs(), "*", out, nameMap);
    } else if (auto divOp = llvm::dyn_cast<arc::DivOp>(&op)) {
      emitDivLikeOp(divOp.getResult(), divOp.getLhs(), divOp.getRhs(), "div",
                    out, nameMap);
    } else if (auto remOp = llvm::dyn_cast<arc::RemOp>(&op)) {
      emitDivLikeOp(remOp.getResult(), remOp.getLhs(), remOp.getRhs(), "mod",
                    out, nameMap);
    } else if (auto cmpOp = llvm::dyn_cast<arc::CmpOp>(&op)) {
      auto lhs = getExpr(cmpOp.getLhs(), nameMap);
      auto rhs = getExpr(cmpOp.getRhs(), nameMap);
      auto pred = cmpOp.getPredicate();
      auto whymlOp = llvm::StringSwitch<llvm::StringRef>(pred)
                         .Case("lt", "<")
                         .Case("le", "<=")
                         .Case("gt", ">")
                         .Case("ge", ">=")
                         .Case("eq", "=")
                         .Case("ne", "<>")
                         .Default("=");
      if (whymlOp == "=" && pred != "eq") {
        llvm::errs() << "warning: unknown comparison predicate '" << pred
                     << "', defaulting to '='\n";
      }
      nameMap[cmpOp.getResult()] =
          "(" + lhs + " " + whymlOp.str() + " " + rhs + ")";
    } else if (auto andOp = llvm::dyn_cast<arc::AndOp>(&op)) {
      auto lhs = getExpr(andOp.getLhs(), nameMap);
      auto rhs = getExpr(andOp.getRhs(), nameMap);
      nameMap[andOp.getResult()] = "(" + lhs + " /\\ " + rhs + ")";
    } else if (auto orOp = llvm::dyn_cast<arc::OrOp>(&op)) {
      auto lhs = getExpr(orOp.getLhs(), nameMap);
      auto rhs = getExpr(orOp.getRhs(), nameMap);
      nameMap[orOp.getResult()] = "(" + lhs + " \\/ " + rhs + ")";
    } else if (auto notOp = llvm::dyn_cast<arc::NotOp>(&op)) {
      auto operand = getExpr(notOp.getOperand(), nameMap);
      nameMap[notOp.getResult()] = "(not " + operand + ")";
    } else if (auto retOp = llvm::dyn_cast<arc::ReturnOp>(&op)) {
      if (retOp.getValue()) {
        auto val = getExpr(retOp.getValue(), nameMap);
        out << "    " << val << "\n";
      }
    } else if (auto varOp = llvm::dyn_cast<arc::VarOp>(&op)) {
      auto init = getExpr(varOp.getInit(), nameMap);
      auto name = varOp.getName().str();
      out << "    let " << name << " = " << init << " in\n";
      nameMap[varOp.getResult()] = name;
    } else if (auto assignOp = llvm::dyn_cast<arc::AssignOp>(&op)) {
      auto value = getExpr(assignOp.getValue(), nameMap);
      // In WhyML, we model reassignment via a let-rebinding.
      // Look up the original variable name from the defining VarOp,
      // rather than the current nameMap expression which may be a
      // compound expression (CR-4: prevents invalid WhyML like
      // "let (a + 1) = ...").
      // Note: getDefiningOp() is O(n) per assignment (walks the def chain).
      // Acceptable for Slice 1 with small functions; future slices should
      // consider a direct VarOp lookup map to avoid quadratic behavior.
      std::string varName;
      if (auto* defOp = assignOp.getTarget().getDefiningOp()) {
        if (auto varOp = llvm::dyn_cast<arc::VarOp>(defOp)) {
          varName = varOp.getName().str();
        }
      }
      if (varName.empty()) {
        // Fallback: use the nameMap expression (may not be a valid
        // identifier, but this path is unlikely in well-formed Slice 1 code).
        varName = getExpr(assignOp.getTarget(), nameMap);
      }
      out << "    let " << varName << " = " << value << " in\n";
      // Update nameMap: the target now refers to the new value expression
      nameMap[assignOp.getTarget()] = value;
    } else if (auto ifOp = llvm::dyn_cast<arc::IfOp>(&op)) {
      auto cond = getExpr(ifOp.getCondition(), nameMap);
      out << "    if " << cond << " then\n";
      // Limitation (Slice 1): nameMap is shared between then/else branches.
      // Mutations in the then branch (e.g., variable declarations or
      // assignments) leak into the else branch.  This mirrors the same
      // limitation in Lowering.cpp (valueMap sharing).  For correct
      // semantics, future slices should copy nameMap before each branch
      // and merge results afterward (phi-node style).  SubsetEnforcer's
      // early-return check partially mitigates this.

      // Emit then region
      if (!ifOp.getThenRegion().empty()) {
        for (auto& thenOp : ifOp.getThenRegion().front().getOperations()) {
          emitOp(thenOp, out, nameMap);
        }
      }
      // Emit else region (always required in WhyML expression syntax)
      if (!ifOp.getElseRegion().empty()) {
        out << "    else\n";
        for (auto& elseOp : ifOp.getElseRegion().front().getOperations()) {
          emitOp(elseOp, out, nameMap);
        }
      } else {
        // WhyML requires an else clause; emit unit for empty else
        out << "    else\n";
        out << "    ()\n";
      }
    }
  }

  // Defensive fallback: "?unknown?" should never appear in well-formed
  // Slice 1 output.  If it does, it indicates a bug in the lowering or
  // emission pipeline (an MLIR value was created but not registered in
  // nameMap).  Not tested directly; covered implicitly by integration tests.
  std::string getExpr(mlir::Value val,
                      llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto it = nameMap.find(val);
    if (it != nameMap.end()) {
      return it->second;
    }
    llvm::errs() << "warning: unmapped MLIR value in WhyML emission\n";
    return "?unknown?";
  }

  mlir::ModuleOp module;
};

} // namespace

std::optional<WhyMLResult> emitWhyML(mlir::ModuleOp module) {
  WhyMLWriter writer(module);
  return writer.emit();
}

} // namespace arcanum
