#include "dialect/Lowering.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"

#include <string>

namespace arcanum {
namespace {

class ArcLowering {
public:
  ArcLowering(mlir::MLIRContext& ctx, clang::ASTContext& astCtx,
              const std::map<const clang::FunctionDecl*, ContractInfo>& contracts)
      : mlirCtx_(ctx), astCtx_(astCtx), contracts_(contracts),
        builder_(&ctx) {
    ctx.getOrLoadDialect<arc::ArcDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> lower() {
    module_ = mlir::ModuleOp::create(builder_.getUnknownLoc());
    builder_.setInsertionPointToEnd(module_->getBody());

    for (auto* decl : astCtx_.getTranslationUnitDecl()->decls()) {
      if (auto* funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
        if (funcDecl->hasBody()) {
          lowerFunction(funcDecl);
        }
      }
    }

    return std::move(module_);
  }

private:
  mlir::Location getLoc(clang::SourceLocation clangLoc) {
    if (clangLoc.isValid()) {
      auto& sm = astCtx_.getSourceManager();
      auto presumed = sm.getPresumedLoc(clangLoc);
      if (presumed.isValid()) {
        return mlir::FileLineColLoc::get(
            builder_.getStringAttr(presumed.getFilename()),
            presumed.getLine(), presumed.getColumn());
      }
    }
    return builder_.getUnknownLoc();
  }

  mlir::Type getArcType(clang::QualType type) {
    auto canonical = type.getCanonicalType();
    if (canonical->isBooleanType()) {
      return arc::BoolType::get(&mlirCtx_);
    }
    // Default to i32 for integer types in Slice 1
    return arc::I32Type::get(&mlirCtx_);
  }

  void lowerFunction(clang::FunctionDecl* funcDecl) {
    auto loc = getLoc(funcDecl->getLocation());
    auto name = funcDecl->getNameAsString();

    // Build function type
    llvm::SmallVector<mlir::Type> paramTypes;
    for (const auto* param : funcDecl->parameters()) {
      paramTypes.push_back(getArcType(param->getType()));
    }
    mlir::Type resultType = getArcType(funcDecl->getReturnType());
    auto funcType = builder_.getFunctionType(paramTypes, {resultType});

    // Get contract strings if present
    mlir::StringAttr requiresAttr, ensuresAttr;
    auto it = contracts_.find(funcDecl);
    if (it != contracts_.end()) {
      // Serialize contract expressions as string attributes for Slice 1.
      // Future slices will use structured MLIR attributes.
      std::string reqStr, ensStr;
      for (size_t i = 0; i < it->second.preconditions.size(); ++i) {
        if (i > 0) reqStr += " && ";
        reqStr += serializeExpr(it->second.preconditions[i]);
      }
      for (size_t i = 0; i < it->second.postconditions.size(); ++i) {
        if (i > 0) ensStr += " && ";
        ensStr += serializeExpr(it->second.postconditions[i]);
      }
      if (!reqStr.empty()) {
        requiresAttr = builder_.getStringAttr(reqStr);
      }
      if (!ensStr.empty()) {
        ensuresAttr = builder_.getStringAttr(ensStr);
      }
    }

    // Create arc.func
    auto funcOp = builder_.create<arc::FuncOp>(
        loc, name, mlir::TypeAttr::get(funcType), requiresAttr, ensuresAttr);

    // Create entry block with parameters
    auto& entryBlock = funcOp.getBody().emplaceBlock();
    for (size_t i = 0; i < paramTypes.size(); ++i) {
      entryBlock.addArgument(paramTypes[i], loc);
    }

    // Map Clang params to MLIR block args
    llvm::DenseMap<const clang::ValueDecl*, mlir::Value> valueMap;
    for (size_t i = 0; i < funcDecl->getNumParams(); ++i) {
      valueMap[funcDecl->getParamDecl(i)] = entryBlock.getArgument(i);
    }

    // Lower function body
    auto savedIp = builder_.saveInsertionPoint();
    builder_.setInsertionPointToEnd(&entryBlock);
    lowerStmt(funcDecl->getBody(), valueMap);
    builder_.restoreInsertionPoint(savedIp);
  }

  void lowerStmt(const clang::Stmt* stmt,
                 llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    if (auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      for (const auto* child : compound->body()) {
        lowerStmt(child, valueMap);
      }
    } else if (auto* ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
      auto retVal = lowerExpr(ret->getRetValue(), valueMap);
      builder_.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()), retVal);
    } else if (auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      for (const auto* d : declStmt->decls()) {
        if (auto* varDecl = llvm::dyn_cast<clang::VarDecl>(d)) {
          if (varDecl->hasInit()) {
            auto initVal = lowerExpr(varDecl->getInit(), valueMap);
            auto loc = getLoc(varDecl->getLocation());
            auto varOp = builder_.create<arc::VarOp>(
                loc, getArcType(varDecl->getType()),
                varDecl->getNameAsString(), initVal);
            valueMap[varDecl] = varOp.getResult();
          }
        }
      }
    } else if (auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
      auto cond = lowerExpr(ifStmt->getCond(), valueMap);
      auto loc = getLoc(ifStmt->getIfLoc());
      auto ifOp = builder_.create<arc::IfOp>(loc, mlir::TypeRange{}, cond);

      // Then region
      auto& thenBlock = ifOp.getThenRegion().emplaceBlock();
      auto savedIp = builder_.saveInsertionPoint();
      builder_.setInsertionPointToEnd(&thenBlock);
      lowerStmt(ifStmt->getThen(), valueMap);
      builder_.restoreInsertionPoint(savedIp);

      // Else region
      if (ifStmt->getElse()) {
        auto& elseBlock = ifOp.getElseRegion().emplaceBlock();
        auto savedIp2 = builder_.saveInsertionPoint();
        builder_.setInsertionPointToEnd(&elseBlock);
        lowerStmt(ifStmt->getElse(), valueMap);
        builder_.restoreInsertionPoint(savedIp2);
      }
    }
    // TODO: handle assignment expressions in Slice 1
  }

  mlir::Value lowerExpr(const clang::Expr* expr,
                        llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    expr = expr->IgnoreParenImpCasts();
    auto loc = getLoc(expr->getBeginLoc());

    if (auto* intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
      auto val = intLit->getValue().getSExtValue();
      return builder_.create<arc::ConstantOp>(
          loc, arc::I32Type::get(&mlirCtx_),
          builder_.getI32IntegerAttr(val));
    }

    if (auto* boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
      return builder_.create<arc::ConstantOp>(
          loc, arc::BoolType::get(&mlirCtx_),
          builder_.getBoolAttr(boolLit->getValue()));
    }

    if (auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
      auto it = valueMap.find(declRef->getDecl());
      if (it != valueMap.end()) {
        return it->second;
      }
      // Fallback: return a zero constant
      return builder_.create<arc::ConstantOp>(
          loc, arc::I32Type::get(&mlirCtx_), builder_.getI32IntegerAttr(0));
    }

    if (auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
      auto lhs = lowerExpr(binOp->getLHS(), valueMap);
      auto rhs = lowerExpr(binOp->getRHS(), valueMap);

      switch (binOp->getOpcode()) {
      case clang::BO_Add:
        return builder_.create<arc::AddOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Sub:
        return builder_.create<arc::SubOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Mul:
        return builder_.create<arc::MulOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Div:
        return builder_.create<arc::DivOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Rem:
        return builder_.create<arc::RemOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_LT:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("lt"), lhs, rhs);
      case clang::BO_LE:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("le"), lhs, rhs);
      case clang::BO_GT:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("gt"), lhs, rhs);
      case clang::BO_GE:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("ge"), lhs, rhs);
      case clang::BO_EQ:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("eq"), lhs, rhs);
      case clang::BO_NE:
        return builder_.create<arc::CmpOp>(
            loc, arc::BoolType::get(&mlirCtx_),
            builder_.getStringAttr("ne"), lhs, rhs);
      case clang::BO_LAnd:
        return builder_.create<arc::AndOp>(
            loc, arc::BoolType::get(&mlirCtx_), lhs, rhs);
      case clang::BO_LOr:
        return builder_.create<arc::OrOp>(
            loc, arc::BoolType::get(&mlirCtx_), lhs, rhs);
      default:
        break;
      }
    }

    if (auto* unaryOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
      auto operand = lowerExpr(unaryOp->getSubExpr(), valueMap);
      switch (unaryOp->getOpcode()) {
      case clang::UO_LNot:
        return builder_.create<arc::NotOp>(
            loc, arc::BoolType::get(&mlirCtx_), operand);
      case clang::UO_Minus: {
        auto zero = builder_.create<arc::ConstantOp>(
            loc, arc::I32Type::get(&mlirCtx_), builder_.getI32IntegerAttr(0));
        return builder_.create<arc::SubOp>(loc, operand.getType(), zero, operand);
      }
      default:
        break;
      }
    }

    // Fallback: return zero constant
    return builder_.create<arc::ConstantOp>(
        loc, arc::I32Type::get(&mlirCtx_), builder_.getI32IntegerAttr(0));
  }

  std::string serializeExpr(const ContractExprPtr& expr) {
    switch (expr->kind) {
    case ContractExprKind::IntLiteral:
      return std::to_string(expr->intValue);
    case ContractExprKind::BoolLiteral:
      return expr->boolValue ? "true" : "false";
    case ContractExprKind::ParamRef:
      return expr->paramName;
    case ContractExprKind::ResultRef:
      return "\\result";
    case ContractExprKind::BinaryOp: {
      std::string op;
      switch (expr->binaryOp) {
      case BinaryOpKind::Add: op = "+"; break;
      case BinaryOpKind::Sub: op = "-"; break;
      case BinaryOpKind::Mul: op = "*"; break;
      case BinaryOpKind::Div: op = "/"; break;
      case BinaryOpKind::Rem: op = "%"; break;
      case BinaryOpKind::Lt: op = "<"; break;
      case BinaryOpKind::Le: op = "<="; break;
      case BinaryOpKind::Gt: op = ">"; break;
      case BinaryOpKind::Ge: op = ">="; break;
      case BinaryOpKind::Eq: op = "=="; break;
      case BinaryOpKind::Ne: op = "!="; break;
      case BinaryOpKind::And: op = "&&"; break;
      case BinaryOpKind::Or: op = "||"; break;
      }
      return "(" + serializeExpr(expr->left) + " " + op + " " +
             serializeExpr(expr->right) + ")";
    }
    case ContractExprKind::UnaryOp: {
      std::string op = expr->unaryOp == UnaryOpKind::Not ? "!" : "-";
      return op + serializeExpr(expr->operand);
    }
    }
    return "";
  }

  mlir::MLIRContext& mlirCtx_;
  clang::ASTContext& astCtx_;
  const std::map<const clang::FunctionDecl*, ContractInfo>& contracts_;
  mlir::OpBuilder builder_;
  mlir::OwningOpRef<mlir::ModuleOp> module_;
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context,
    clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts) {
  ArcLowering lowering(context, astContext, contracts);
  return lowering.lower();
}

} // namespace arcanum
