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

/// Get the minimum value of an IntType as a decimal string.
std::string getMinStr(arc::IntType type) {
  llvm::APInt minVal = type.getMinValue();
  llvm::SmallString<32> str;
  if (type.getIsSigned()) {
    minVal.toStringSigned(str);
  } else {
    minVal.toStringUnsigned(str);
  }
  return std::string(str);
}

/// Get the maximum value of an IntType as a decimal string.
std::string getMaxStr(arc::IntType type) {
  llvm::APInt maxVal = type.getMaxValue();
  llvm::SmallString<32> str;
  if (type.getIsSigned()) {
    maxVal.toStringSigned(str);
  } else {
    maxVal.toStringUnsigned(str);
  }
  return std::string(str);
}

/// Get 2^N as a decimal string (for modular arithmetic).
std::string getPowerOfTwo(unsigned width) {
  llvm::APInt val = llvm::APInt(width + 1, 1).shl(width);
  llvm::SmallString<32> str;
  val.toStringUnsigned(str);
  return std::string(str);
}

/// Get 2^(N-1) as a decimal string (for signed wrap offset).
std::string getHalfPowerOfTwo(unsigned width) {
  llvm::APInt val = llvm::APInt(width, 1).shl(width - 1);
  llvm::SmallString<32> str;
  val.toStringUnsigned(str);
  return std::string(str);
}

/// Read the overflow mode from an operation's "overflow" string attribute.
/// Returns "trap" if no attribute is present (default mode).
llvm::StringRef getOverflowMode(mlir::Operation* op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("overflow")) {
    return attr.getValue();
  }
  return "trap";
}

/// Extract the IntType from an MLIR operation's result type.
/// Returns nullptr if the result is not an arc::IntType.
arc::IntType getResultIntType(mlir::Operation* op) {
  if (op->getNumResults() > 0) {
    return mlir::dyn_cast<arc::IntType>(op->getResult(0).getType());
  }
  return {};
}

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

/// Convert snake_case to CamelCase (capitalize each word after '_').
std::string snakeToCamelCase(llvm::StringRef name) {
  std::string camel;
  bool nextUpper = true;
  for (char c : name) {
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

/// Convert a CamelCase or snake_case function name to a WhyML module name
/// (first letter capitalized).
std::string toModuleName(llvm::StringRef funcName) {
  if (funcName.empty()) {
    return "Module";
  }
  return snakeToCamelCase(funcName);
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

/// Build a saturate (clamp) expression for the given raw expression and type.
/// Returns: (if expr < MIN then MIN else if expr > MAX then MAX else expr)
std::string buildSaturateExpr(const std::string& expr, arc::IntType intType) {
  auto minStr = getMinStr(intType);
  auto maxStr = getMaxStr(intType);
  return "(if " + expr + " < " + minStr + " then " + minStr + " else if " +
         expr + " > " + maxStr + " then " + maxStr + " else " + expr + ")";
}

/// Apply the overflow mode to a raw arithmetic expression, returning the
/// mode-adjusted expression. For "trap" mode, emits an assertion to `out`
/// and returns the raw expression unchanged. For "wrap", returns the modular
/// reduction expression. For "saturate", returns the clamp expression.
std::string applyOverflowMode(const std::string& expr, arc::IntType intType,
                              llvm::StringRef mode,
                              llvm::raw_string_ostream& out) {
  if (mode == "wrap") {
    if (intType && !intType.getIsSigned()) {
      return "(mod " + expr + " " + getPowerOfTwo(intType.getWidth()) + ")";
    }
    if (intType) {
      auto halfPow = getHalfPowerOfTwo(intType.getWidth());
      auto fullPow = getPowerOfTwo(intType.getWidth());
      auto shifted = "(" + expr + " + " + halfPow + ")";
      auto modded = "(mod " + shifted + " " + fullPow + ")";
      return "(" + modded + " - " + halfPow + ")";
    }
    return expr;
  }
  if (mode == "saturate") {
    if (!intType) {
      llvm::errs() << "warning: saturate mode on op without IntType; "
                      "falling back to trap semantics\n";
      return expr;
    }
    return buildSaturateExpr(expr, intType);
  }
  // Trap mode (default): assert bounds
  if (intType) {
    out << "    assert { " << getMinStr(intType) << " <= " << expr << " /\\ "
        << expr << " <= " << getMaxStr(intType) << " };\n";
  }
  return expr;
}

/// Scan a FuncOp to determine which Why3 modules are needed.
struct ModuleImports {
  bool needsBool = false;
  bool needsComputerDivision = false;
};

ModuleImports computeModuleImports(arc::FuncOp funcOp) {
  ModuleImports imports;
  auto funcType = funcOp.getFunctionType();
  auto& entryBlock = funcOp.getBody().front();

  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    if (mlir::isa<arc::BoolType>(entryBlock.getArgument(i).getType())) {
      imports.needsBool = true;
    }
  }
  if (funcType.getNumResults() > 0 &&
      mlir::isa<arc::BoolType>(funcType.getResult(0))) {
    imports.needsBool = true;
  }
  // Scan for DivOp/RemOp or wrap-mode arithmetic/casts to determine if
  // ComputerDivision is needed (wrap mode uses mod for modular arithmetic)
  funcOp.walk([&](mlir::Operation* op) {
    if (llvm::isa<arc::DivOp>(op) || llvm::isa<arc::RemOp>(op)) {
      imports.needsComputerDivision = true;
    }
    if (llvm::isa<arc::AddOp, arc::SubOp, arc::MulOp>(op)) {
      if (getOverflowMode(op) == "wrap") {
        imports.needsComputerDivision = true;
      }
    }
    // CastOp with wrap mode also uses mod for modular reduction
    if (llvm::isa<arc::CastOp>(op)) {
      if (getOverflowMode(op) == "wrap") {
        imports.needsComputerDivision = true;
      }
    }
  });
  return imports;
}

/// Write the module header (module name + use directives) to `out`.
void emitModuleHeader(llvm::raw_string_ostream& out,
                      const std::string& moduleName,
                      const ModuleImports& imports) {
  out << "module " << moduleName << "\n";
  out << "  use int.Int\n";
  if (imports.needsBool) {
    out << "  use bool.Bool\n";
  }
  if (imports.needsComputerDivision) {
    out << "  use int.ComputerDivision\n";
  }
  out << "\n";
}

/// Write the function signature line (let funcName (params): retType) to `out`.
void emitFunctionSignature(llvm::raw_string_ostream& out, arc::FuncOp funcOp,
                           const std::string& funcName) {
  auto& entryBlock = funcOp.getBody().front();
  auto funcType = funcOp.getFunctionType();
  auto paramNames = getParamNames(funcOp);

  out << "  let " << funcName << " ";
  for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
    std::string pname =
        (i < paramNames.size()) ? paramNames[i] : ("arg" + std::to_string(i));
    out << "(" << pname << ": ";
    out << (mlir::isa<arc::BoolType>(entryBlock.getArgument(i).getType())
                ? "bool"
                : "int");
    out << ") ";
  }
  std::string resultType = "int";
  if (funcType.getNumResults() > 0 &&
      mlir::isa<arc::BoolType>(funcType.getResult(0))) {
    resultType = "bool";
  }
  out << ": " << resultType << "\n";
}

/// Write requires/ensures contract clauses to `out`.
void emitContractClauses(llvm::raw_string_ostream& out, arc::FuncOp funcOp) {
  if (auto reqAttr = funcOp.getRequiresAttrAttr()) {
    out << "    requires { " << contractToWhyML(reqAttr.getValue()) << " }\n";
  }
  if (auto ensAttr = funcOp.getEnsuresAttrAttr()) {
    out << "    ensures  { " << contractToWhyML(ensAttr.getValue()) << " }\n";
  }
}

/// Populate the location and module maps in `result` for a given FuncOp.
void populateLocationMaps(arc::FuncOp funcOp, const std::string& funcName,
                          const std::string& moduleName, WhyMLResult& result) {
  LocationEntry entry;
  entry.functionName = funcName;
  auto loc = funcOp.getLoc();
  if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
    entry.fileName = fileLoc.getFilename().str();
    entry.line = fileLoc.getLine();
  }
  result.locationMap[funcName] = entry;
  result.moduleToFuncMap[moduleName] = funcName;
}

/// Write the .mlw text to a temporary file and return the file path.
/// Returns std::nullopt on I/O failure.
std::optional<std::string> writeToTempFile(const std::string& text) {
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
  out << text;
  out.close();
  if (out.has_error()) {
    out.clear_error();
    return std::nullopt;
  }
  return tmpPath.str().str();
}

/// Map a comparison predicate string to the corresponding WhyML operator.
/// Returns "=" as a default and emits a warning for unknown predicates.
llvm::StringRef predicateToWhyMLOp(llvm::StringRef pred) {
  return llvm::StringSwitch<llvm::StringRef>(pred)
      .Case("lt", "<")
      .Case("le", "<=")
      .Case("gt", ">")
      .Case("ge", ">=")
      .Case("eq", "=")
      .Case("ne", "<>")
      .Default("=");
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

    auto filePath = writeToTempFile(result.whymlText);
    if (!filePath) {
      return std::nullopt;
    }
    result.filePath = *filePath;
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

    auto imports = computeModuleImports(funcOp);
    emitModuleHeader(out, moduleName, imports);
    emitFunctionSignature(out, funcOp, funcName);
    emitContractClauses(out, funcOp);

    out << "  =\n";
    emitBody(funcOp, out);
    out << "\nend\n\n";

    result.whymlText += outBuf;
    populateLocationMaps(funcOp, funcName, moduleName, result);
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

  /// Emit a type-aware overflow assertion: assert { MIN <= expr /\ expr <= MAX
  /// }
  void emitTrapAssertion(const std::string& expr, arc::IntType intType,
                         llvm::raw_string_ostream& out) {
    out << "    assert { " << getMinStr(intType) << " <= " << expr << " /\\ "
        << expr << " <= " << getMaxStr(intType) << " };\n";
  }

  /// Emit a binary arithmetic op with mode-aware overflow handling.
  void emitArithWithOverflowCheck(
      mlir::Operation* op, mlir::Value lhsVal, mlir::Value rhsVal,
      const std::string& whymlOp, llvm::raw_string_ostream& out,
      llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto lhs = getExpr(lhsVal, nameMap);
    auto rhs = getExpr(rhsVal, nameMap);
    auto rawExpr = "(" + lhs + " " + whymlOp + " " + rhs + ")";
    auto intType = getResultIntType(op);
    auto mode = getOverflowMode(op);
    nameMap[op->getResult(0)] = applyOverflowMode(rawExpr, intType, mode, out);
  }

  /// Emit a division-like op with divisor-not-zero and overflow assertions.
  /// Division-by-zero assertion is always emitted (undefined in all modes).
  /// Overflow assertion (e.g., MIN / -1) respects the overflow mode:
  /// trap = assert range, wrap = modular reduction, saturate = clamp.
  void emitDivLikeOp(mlir::Operation* op, mlir::Value lhsVal,
                     mlir::Value rhsVal, const std::string& whymlFunc,
                     llvm::raw_string_ostream& out,
                     llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto lhs = getExpr(lhsVal, nameMap);
    auto rhs = getExpr(rhsVal, nameMap);
    auto expr = "(" + whymlFunc + " " + lhs + " " + rhs + ")";
    // Division-by-zero is always undefined, regardless of overflow mode.
    out << "    assert { " << rhs << " <> 0 };\n";
    auto intType = getResultIntType(op);
    auto mode = getOverflowMode(op);
    nameMap[op->getResult(0)] = applyOverflowMode(expr, intType, mode, out);
  }

  /// Emit a CastOp: widening is identity, narrowing/sign-change respects
  /// the overflow mode (trap = assert range, wrap = modular, saturate = clamp).
  void emitCastOp(arc::CastOp castOp, llvm::raw_string_ostream& out,
                  llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto inputExpr = getExpr(castOp.getInput(), nameMap);
    auto srcType = mlir::dyn_cast<arc::IntType>(castOp.getInput().getType());
    auto dstType = mlir::dyn_cast<arc::IntType>(castOp.getResult().getType());

    if (srcType && dstType) {
      // A cast is widening (no assertion needed) only when:
      // 1. Same signedness and destination strictly wider, OR
      // 2. Unsigned-to-signed and destination strictly wider (not same width,
      //    since e.g. u32 max 4294967295 > i32 max 2147483647).
      bool isWidening = dstType.getWidth() > srcType.getWidth() &&
                        (dstType.getIsSigned() == srcType.getIsSigned() ||
                         (!srcType.getIsSigned() && dstType.getIsSigned()));
      if (!isWidening) {
        auto mode = getOverflowMode(castOp);
        inputExpr = applyOverflowMode(inputExpr, dstType, mode, out);
      }
    }
    // The cast itself is an identity in WhyML (all integers are mathematical)
    nameMap[castOp.getResult()] = inputExpr;
  }

  /// Emit a ConstantOp: format the APInt value and register in nameMap.
  void emitConstantOp(arc::ConstantOp constOp,
                      llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto attr = constOp.getValue();
    std::string valStr;
    if (auto intAttr = llvm::dyn_cast<mlir::IntegerAttr>(attr)) {
      // Use APInt directly to correctly handle unsigned 64-bit values
      // that exceed INT64_MAX (e.g., UINT64_MAX).
      llvm::APInt apVal = intAttr.getValue();
      llvm::SmallString<32> valBuf;
      auto resIntType = getResultIntType(constOp);
      if (resIntType && !resIntType.getIsSigned()) {
        apVal.toStringUnsigned(valBuf);
      } else {
        apVal.toStringSigned(valBuf);
      }
      valStr = std::string(valBuf);
    } else if (auto boolAttr = llvm::dyn_cast<mlir::BoolAttr>(attr)) {
      valStr = boolAttr.getValue() ? "true" : "false";
    }
    nameMap[constOp.getResult()] = valStr;
  }

  /// Emit a CmpOp: map predicate to WhyML operator and register in nameMap.
  void emitCmpOp(arc::CmpOp cmpOp,
                 llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto lhs = getExpr(cmpOp.getLhs(), nameMap);
    auto rhs = getExpr(cmpOp.getRhs(), nameMap);
    auto pred = cmpOp.getPredicate();
    auto whymlOp = predicateToWhyMLOp(pred);
    if (whymlOp == "=" && pred != "eq") {
      llvm::errs() << "warning: unknown comparison predicate '" << pred
                   << "', defaulting to '='\n";
    }
    nameMap[cmpOp.getResult()] =
        "(" + lhs + " " + whymlOp.str() + " " + rhs + ")";
  }

  /// Emit an IfOp: condition, then/else regions, empty-else fallback.
  void emitIfOp(arc::IfOp ifOp, llvm::raw_string_ostream& out,
                llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto cond = getExpr(ifOp.getCondition(), nameMap);
    out << "    if " << cond << " then\n";
    // Limitation (Slice 1): nameMap is shared between then/else branches.
    // Mutations in the then branch (e.g., variable declarations or
    // assignments) leak into the else branch.  This mirrors the same
    // limitation in Lowering.cpp (valueMap sharing).  For correct
    // semantics, future slices should copy nameMap before each branch
    // and merge results afterward (phi-node style).  SubsetEnforcer's
    // early-return check partially mitigates this.
    if (!ifOp.getThenRegion().empty()) {
      for (auto& thenOp : ifOp.getThenRegion().front().getOperations()) {
        emitOp(thenOp, out, nameMap);
      }
    }
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

  void emitOp(mlir::Operation& op, llvm::raw_string_ostream& out,
              llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    if (auto constOp = llvm::dyn_cast<arc::ConstantOp>(&op)) {
      emitConstantOp(constOp, nameMap);
    } else if (auto addOp = llvm::dyn_cast<arc::AddOp>(&op)) {
      emitArithWithOverflowCheck(&op, addOp.getLhs(), addOp.getRhs(), "+", out,
                                 nameMap);
    } else if (auto subOp = llvm::dyn_cast<arc::SubOp>(&op)) {
      emitArithWithOverflowCheck(&op, subOp.getLhs(), subOp.getRhs(), "-", out,
                                 nameMap);
    } else if (auto mulOp = llvm::dyn_cast<arc::MulOp>(&op)) {
      emitArithWithOverflowCheck(&op, mulOp.getLhs(), mulOp.getRhs(), "*", out,
                                 nameMap);
    } else if (auto divOp = llvm::dyn_cast<arc::DivOp>(&op)) {
      emitDivLikeOp(&op, divOp.getLhs(), divOp.getRhs(), "div", out, nameMap);
    } else if (auto remOp = llvm::dyn_cast<arc::RemOp>(&op)) {
      emitDivLikeOp(&op, remOp.getLhs(), remOp.getRhs(), "mod", out, nameMap);
    } else if (auto castOp = llvm::dyn_cast<arc::CastOp>(&op)) {
      emitCastOp(castOp, out, nameMap);
    } else if (auto cmpOp = llvm::dyn_cast<arc::CmpOp>(&op)) {
      emitCmpOp(cmpOp, nameMap);
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
      emitIfOp(ifOp, out, nameMap);
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
