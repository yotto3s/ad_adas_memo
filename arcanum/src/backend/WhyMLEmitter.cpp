#include "backend/WhyMLEmitter.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <fstream>
#include <sstream>

namespace arcanum {
namespace {

/// Convert a contract expression string from Arc format to WhyML format.
/// Transforms: \result -> result, && -> /\, || -> \/, etc.
std::string contractToWhyML(llvm::StringRef contract) {
  std::string result;
  size_t i = 0;
  while (i < contract.size()) {
    if (contract.substr(i).starts_with("\\result")) {
      result += "result";
      i += 7;
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
    result[0] = std::toupper(result[0]);
    // Convert snake_case to CamelCase for module name
    std::string camel;
    bool nextUpper = true;
    for (char c : result) {
      if (c == '_') {
        nextUpper = true;
      } else if (nextUpper) {
        camel += std::toupper(c);
        nextUpper = false;
      } else {
        camel += c;
      }
    }
    return camel;
  }
  return "Module";
}

class WhyMLWriter {
public:
  explicit WhyMLWriter(mlir::ModuleOp module) : module_(module) {}

  std::optional<WhyMLResult> emit() {
    WhyMLResult result;

    module_.walk([&](arc::FuncOp funcOp) {
      emitFunction(funcOp, result);
    });

    if (result.whymlText.empty()) {
      return std::nullopt;
    }

    // Write to temp file
    llvm::SmallString<128> tmpPath;
    std::error_code ec;
    ec = llvm::sys::fs::createTemporaryFile("arcanum", "mlw", tmpPath);
    if (ec) {
      return std::nullopt;
    }

    std::ofstream out(tmpPath.c_str());
    out << result.whymlText;
    out.close();

    result.filePath = tmpPath.str().str();
    return result;
  }

private:
  void emitFunction(arc::FuncOp funcOp, WhyMLResult& result) {
    std::ostringstream out;
    auto funcName = funcOp.getSymName().str();
    auto moduleName = toModuleName(funcName);

    out << "module " << moduleName << "\n";
    out << "  use int.Int\n\n";

    // Function signature
    auto funcType = funcOp.getFunctionType();
    out << "  let " << funcName << " ";

    // Parameters
    auto& entryBlock = funcOp.getBody().front();
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      // Try to get parameter name from source location or use default
      out << "(arg" << i << ": int) ";
    }
    out << ": int\n";

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

    result.whymlText += out.str();

    // Populate location map
    auto loc = funcOp.getLoc();
    LocationEntry entry;
    entry.functionName = funcName;
    if (auto fileLoc = llvm::dyn_cast<mlir::FileLineColLoc>(loc)) {
      entry.fileName = fileLoc.getFilename().str();
      entry.line = fileLoc.getLine();
    }
    result.locationMap[funcName] = entry;
  }

  void emitBody(arc::FuncOp funcOp, std::ostringstream& out) {
    auto& entryBlock = funcOp.getBody().front();
    // Map MLIR values to WhyML variable names
    llvm::DenseMap<mlir::Value, std::string> nameMap;

    // Map block arguments to parameter names
    for (unsigned i = 0; i < entryBlock.getNumArguments(); ++i) {
      nameMap[entryBlock.getArgument(i)] = "arg" + std::to_string(i);
    }

    for (auto& op : entryBlock.getOperations()) {
      emitOp(op, out, nameMap);
    }
  }

  void emitOp(mlir::Operation& op, std::ostringstream& out,
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
      auto lhs = getExpr(addOp.getLhs(), nameMap);
      auto rhs = getExpr(addOp.getRhs(), nameMap);
      auto expr = "(" + lhs + " + " + rhs + ")";

      // Emit overflow assertion for trap mode
      out << "    (* overflow check for addition *)\n";
      out << "    assert { -2147483648 <= " << lhs << " + " << rhs
          << " <= 2147483647 };\n";

      nameMap[addOp.getResult()] = expr;
    } else if (auto subOp = llvm::dyn_cast<arc::SubOp>(&op)) {
      auto lhs = getExpr(subOp.getLhs(), nameMap);
      auto rhs = getExpr(subOp.getRhs(), nameMap);
      auto expr = "(" + lhs + " - " + rhs + ")";
      out << "    assert { -2147483648 <= " << lhs << " - " << rhs
          << " <= 2147483647 };\n";
      nameMap[subOp.getResult()] = expr;
    } else if (auto mulOp = llvm::dyn_cast<arc::MulOp>(&op)) {
      auto lhs = getExpr(mulOp.getLhs(), nameMap);
      auto rhs = getExpr(mulOp.getRhs(), nameMap);
      auto expr = "(" + lhs + " * " + rhs + ")";
      out << "    assert { -2147483648 <= " << lhs << " * " << rhs
          << " <= 2147483647 };\n";
      nameMap[mulOp.getResult()] = expr;
    } else if (auto divOp = llvm::dyn_cast<arc::DivOp>(&op)) {
      auto lhs = getExpr(divOp.getLhs(), nameMap);
      auto rhs = getExpr(divOp.getRhs(), nameMap);
      auto expr = "(div " + lhs + " " + rhs + ")";
      out << "    assert { " << rhs << " <> 0 };\n";
      nameMap[divOp.getResult()] = expr;
    } else if (auto remOp = llvm::dyn_cast<arc::RemOp>(&op)) {
      auto lhs = getExpr(remOp.getLhs(), nameMap);
      auto rhs = getExpr(remOp.getRhs(), nameMap);
      auto expr = "(mod " + lhs + " " + rhs + ")";
      out << "    assert { " << rhs << " <> 0 };\n";
      nameMap[remOp.getResult()] = expr;
    } else if (auto cmpOp = llvm::dyn_cast<arc::CmpOp>(&op)) {
      auto lhs = getExpr(cmpOp.getLhs(), nameMap);
      auto rhs = getExpr(cmpOp.getRhs(), nameMap);
      auto pred = cmpOp.getPredicate().str();
      std::string whymlOp;
      if (pred == "lt") whymlOp = "<";
      else if (pred == "le") whymlOp = "<=";
      else if (pred == "gt") whymlOp = ">";
      else if (pred == "ge") whymlOp = ">=";
      else if (pred == "eq") whymlOp = "=";
      else if (pred == "ne") whymlOp = "<>";
      nameMap[cmpOp.getResult()] = "(" + lhs + " " + whymlOp + " " + rhs + ")";
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
    }
    // Note: IfOp handling is more complex and will emit if-then-else in WhyML
  }

  std::string getExpr(mlir::Value val,
                      llvm::DenseMap<mlir::Value, std::string>& nameMap) {
    auto it = nameMap.find(val);
    if (it != nameMap.end()) {
      return it->second;
    }
    return "?unknown?";
  }

  mlir::ModuleOp module_;
};

} // namespace

std::optional<WhyMLResult> emitWhyML(mlir::ModuleOp module) {
  WhyMLWriter writer(module);
  return writer.emit();
}

} // namespace arcanum
