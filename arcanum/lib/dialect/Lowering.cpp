#include "dialect/Lowering.h"
#include "DiagnosticTracker.h"
#include "dialect/ArcDialect.h"
#include "dialect/ArcOps.h"
#include "dialect/ArcTypes.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace arcanum {
namespace {

class ArcLowering {
public:
  ArcLowering(
      mlir::MLIRContext& ctx, clang::ASTContext& astCtx,
      const std::map<const clang::FunctionDecl*, ContractInfo>& contracts)
      : mlirCtx(ctx), astCtx(astCtx), contracts(contracts), builder(&ctx) {
    ctx.getOrLoadDialect<arc::ArcDialect>();
  }

  mlir::OwningOpRef<mlir::ModuleOp> lower() {
    module = mlir::ModuleOp::create(builder.getUnknownLoc());
    builder.setInsertionPointToEnd(module->getBody());

    for (auto* decl : astCtx.getTranslationUnitDecl()->decls()) {
      if (auto* funcDecl = llvm::dyn_cast<clang::FunctionDecl>(decl)) {
        if (funcDecl->hasBody()) {
          lowerFunction(funcDecl);
        }
      }
    }

    return std::move(module);
  }

private:
  mlir::Location getLoc(clang::SourceLocation clangLoc) {
    if (clangLoc.isValid()) {
      auto& sm = astCtx.getSourceManager();
      auto presumed = sm.getPresumedLoc(clangLoc);
      if (presumed.isValid()) {
        return mlir::FileLineColLoc::get(
            builder.getStringAttr(presumed.getFilename()), presumed.getLine(),
            presumed.getColumn());
      }
    }
    return builder.getUnknownLoc();
  }

  mlir::Type getArcType(clang::QualType type) {
    auto canonical = type.getCanonicalType();
    if (canonical->isVoidType()) {
      // Void type should not appear in Slice 1 (SubsetEnforcer allows void
      // return types, but they produce no return value).  Map to i32 as a
      // conservative fallback; future slices will handle void properly.
      llvm::errs()
          << "warning: void type mapped to i32 in getArcType fallback\n";
      DiagnosticTracker::recordFallback();
      return arc::I32Type::get(&mlirCtx);
    }
    if (canonical->isBooleanType()) {
      return arc::BoolType::get(&mlirCtx);
    }
    // Default to i32 for integer types in Slice 1
    return arc::I32Type::get(&mlirCtx);
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
    auto funcType = builder.getFunctionType(paramTypes, {resultType});

    // Get contract strings if present
    mlir::StringAttr requiresAttr;
    mlir::StringAttr ensuresAttr;
    auto it = contracts.find(funcDecl);
    if (it != contracts.end()) {
      // Serialize contract expressions as string attributes for Slice 1.
      // Future slices will use structured MLIR attributes.
      std::string reqStr;
      std::string ensStr;
      for (size_t i = 0; i < it->second.preconditions.size(); ++i) {
        if (i > 0) {
          reqStr += " && ";
        }
        reqStr += serializeExpr(it->second.preconditions[i]);
      }
      for (size_t i = 0; i < it->second.postconditions.size(); ++i) {
        if (i > 0) {
          ensStr += " && ";
        }
        ensStr += serializeExpr(it->second.postconditions[i]);
      }
      if (!reqStr.empty()) {
        requiresAttr = builder.getStringAttr(reqStr);
      }
      if (!ensStr.empty()) {
        ensuresAttr = builder.getStringAttr(ensStr);
      }
    }

    // Create arc.func
    auto funcOp = builder.create<arc::FuncOp>(loc, builder.getStringAttr(name),
                                              mlir::TypeAttr::get(funcType),
                                              requiresAttr, ensuresAttr);

    // Store parameter names as an attribute for the WhyML emitter
    llvm::SmallVector<mlir::Attribute> paramNameAttrs;
    for (const auto* param : funcDecl->parameters()) {
      paramNameAttrs.push_back(builder.getStringAttr(param->getNameAsString()));
    }
    funcOp->setAttr("param_names", builder.getArrayAttr(paramNameAttrs));

    // Create entry block with parameters
    auto& entryBlock = funcOp.getBody().emplaceBlock();
    for (auto paramType : paramTypes) {
      entryBlock.addArgument(paramType, loc);
    }

    // Map Clang params to MLIR block args
    llvm::DenseMap<const clang::ValueDecl*, mlir::Value> valueMap;
    for (size_t i = 0; i < funcDecl->getNumParams(); ++i) {
      valueMap[funcDecl->getParamDecl(i)] = entryBlock.getArgument(i);
    }

    // Lower function body
    auto savedIp = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&entryBlock);
    lowerStmt(funcDecl->getBody(), valueMap);
    builder.restoreInsertionPoint(savedIp);
  }

  void
  lowerStmt(const clang::Stmt* stmt,
            llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      for (const auto* child : compound->body()) {
        lowerStmt(child, valueMap);
      }
    } else if (const auto* ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
      auto retVal = lowerExpr(ret->getRetValue(), valueMap);
      builder.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()), retVal);
    } else if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      for (const auto* d : declStmt->decls()) {
        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(d)) {
          if (varDecl->hasInit()) {
            auto initVal = lowerExpr(varDecl->getInit(), valueMap);
            auto loc = getLoc(varDecl->getLocation());
            auto varOp =
                builder.create<arc::VarOp>(loc, getArcType(varDecl->getType()),
                                           varDecl->getNameAsString(), initVal);
            valueMap[varDecl] = varOp.getResult();
          }
        }
      }
    } else if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
      // Limitation (Slice 1): IfOp lowering does not propagate return values
      // out of if/else branches.  Mutations to valueMap inside then/else
      // regions may not be visible after the IfOp.  Combined with
      // unimplemented assignment lowering, non-terminal if/else (i.e.,
      // if/else that is not the last statement with returns in both
      // branches) may produce incorrect MLIR.  SubsetEnforcer's early-return
      // check partially mitigates this by rejecting guard-clause patterns.
      auto cond = lowerExpr(ifStmt->getCond(), valueMap);
      auto loc = getLoc(ifStmt->getIfLoc());
      auto ifOp = builder.create<arc::IfOp>(loc, mlir::TypeRange{}, cond);

      // Then region
      auto& thenBlock = ifOp.getThenRegion().emplaceBlock();
      auto savedIp = builder.saveInsertionPoint();
      builder.setInsertionPointToEnd(&thenBlock);
      lowerStmt(ifStmt->getThen(), valueMap);
      builder.restoreInsertionPoint(savedIp);

      // Else region
      if (ifStmt->getElse() != nullptr) {
        auto& elseBlock = ifOp.getElseRegion().emplaceBlock();
        auto savedIp2 = builder.saveInsertionPoint();
        builder.setInsertionPointToEnd(&elseBlock);
        lowerStmt(ifStmt->getElse(), valueMap);
        builder.restoreInsertionPoint(savedIp2);
      }
    }
    // TODO: handle assignment expressions in Slice 1
  }

  mlir::Value
  lowerExpr(const clang::Expr* expr,
            llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    expr = expr->IgnoreParenImpCasts();
    auto loc = getLoc(expr->getBeginLoc());

    if (const auto* intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
      auto val = intLit->getValue().getSExtValue();
      return builder.create<arc::ConstantOp>(
          loc, arc::I32Type::get(&mlirCtx),
          builder.getI32IntegerAttr(static_cast<int32_t>(val)));
    }

    if (const auto* boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
      return builder.create<arc::ConstantOp>(
          loc, arc::BoolType::get(&mlirCtx),
          builder.getBoolAttr(boolLit->getValue()));
    }

    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
      auto it = valueMap.find(declRef->getDecl());
      if (it != valueMap.end()) {
        return it->second;
      }
      // Fallback: return a zero constant with diagnostic
      llvm::errs() << "warning: unknown declaration reference '"
                   << declRef->getDecl()->getNameAsString()
                   << "', using zero fallback\n";
      DiagnosticTracker::recordFallback();
      return builder.create<arc::ConstantOp>(loc, arc::I32Type::get(&mlirCtx),
                                             builder.getI32IntegerAttr(0));
    }

    if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
      auto lhs = lowerExpr(binOp->getLHS(), valueMap);
      auto rhs = lowerExpr(binOp->getRHS(), valueMap);

      switch (binOp->getOpcode()) {
      case clang::BO_Add:
        return builder.create<arc::AddOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Sub:
        return builder.create<arc::SubOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Mul:
        return builder.create<arc::MulOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Div:
        return builder.create<arc::DivOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_Rem:
        return builder.create<arc::RemOp>(loc, lhs.getType(), lhs, rhs);
      case clang::BO_LT:
        return builder.create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                          builder.getStringAttr("lt"), lhs,
                                          rhs);
      case clang::BO_LE:
        return builder.create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                          builder.getStringAttr("le"), lhs,
                                          rhs);
      case clang::BO_GT:
        return builder.create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                          builder.getStringAttr("gt"), lhs,
                                          rhs);
      case clang::BO_GE:
        return builder.create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                          builder.getStringAttr("ge"), lhs,
                                          rhs);
      case clang::BO_EQ:
        return builder.create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                          builder.getStringAttr("eq"), lhs,
                                          rhs);
      case clang::BO_NE:
        return builder.create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                          builder.getStringAttr("ne"), lhs,
                                          rhs);
      case clang::BO_LAnd:
        return builder.create<arc::AndOp>(loc, arc::BoolType::get(&mlirCtx),
                                          lhs, rhs);
      case clang::BO_LOr:
        return builder.create<arc::OrOp>(loc, arc::BoolType::get(&mlirCtx), lhs,
                                         rhs);
      default:
        llvm::errs() << "warning: unhandled binary operator opcode "
                     << binOp->getOpcodeStr() << "\n";
        break;
      }
    }

    if (const auto* unaryOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
      auto operand = lowerExpr(unaryOp->getSubExpr(), valueMap);
      switch (unaryOp->getOpcode()) {
      case clang::UO_LNot:
        return builder.create<arc::NotOp>(loc, arc::BoolType::get(&mlirCtx),
                                          operand);
      case clang::UO_Minus: {
        auto zero = builder.create<arc::ConstantOp>(
            loc, arc::I32Type::get(&mlirCtx), builder.getI32IntegerAttr(0));
        return builder.create<arc::SubOp>(loc, operand.getType(), zero,
                                          operand);
      }
      default:
        break;
      }
    }

    // Fallback: return zero constant with diagnostic
    llvm::errs() << "warning: unrecognized expression in lowering, using zero "
                    "fallback\n";
    DiagnosticTracker::recordFallback();
    return builder.create<arc::ConstantOp>(loc, arc::I32Type::get(&mlirCtx),
                                           builder.getI32IntegerAttr(0));
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
      case BinaryOpKind::Add:
        op = "+";
        break;
      case BinaryOpKind::Sub:
        op = "-";
        break;
      case BinaryOpKind::Mul:
        op = "*";
        break;
      case BinaryOpKind::Div:
        op = "/";
        break;
      case BinaryOpKind::Rem:
        op = "%";
        break;
      case BinaryOpKind::Lt:
        op = "<";
        break;
      case BinaryOpKind::Le:
        op = "<=";
        break;
      case BinaryOpKind::Gt:
        op = ">";
        break;
      case BinaryOpKind::Ge:
        op = ">=";
        break;
      case BinaryOpKind::Eq:
        op = "==";
        break;
      case BinaryOpKind::Ne:
        op = "!=";
        break;
      case BinaryOpKind::And:
        op = "&&";
        break;
      case BinaryOpKind::Or:
        op = "||";
        break;
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

  mlir::MLIRContext& mlirCtx;
  clang::ASTContext& astCtx;
  const std::map<const clang::FunctionDecl*, ContractInfo>& contracts;
  mlir::OpBuilder builder;
  mlir::OwningOpRef<mlir::ModuleOp> module;
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context, clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts) {
  ArcLowering lowering(context, astContext, contracts);
  return lowering.lower();
}

} // namespace arcanum
