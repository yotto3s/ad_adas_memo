#include "arcanum/dialect/Lowering.h"
#include "arcanum/DiagnosticTracker.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"

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

#include <array>
#include <optional>
#include <string>

namespace arcanum {

/// Lookup table mapping BinaryOpKind to its operator string for serialization.
static constexpr std::array<const char*, 13> BINARY_OP_STRINGS = {
    "+",  // Add
    "-",  // Sub
    "*",  // Mul
    "/",  // Div
    "%",  // Rem
    "<",  // Lt
    "<=", // Le
    ">",  // Gt
    ">=", // Ge
    "==", // Eq
    "!=", // Ne
    "&&", // And
    "||", // Or
};
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

    // Iterate over top-level TU declarations only.
    // IMPORTANT: This must iterate the same declaration list as
    // parseContracts() in ContractParser.cpp, because the contract map
    // is keyed by FunctionDecl pointer identity.  Both functions must
    // agree on which declarations they process.  SubsetEnforcer also
    // validates that only TU-level functions are accepted.
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
      llvm::errs()
          << "warning: void type mapped to i32 in getArcType fallback\n";
      DiagnosticTracker::recordFallback();
      return arc::IntType::get(&mlirCtx, 32, true);
    }
    if (canonical->isBooleanType()) {
      return arc::BoolType::get(&mlirCtx);
    }
    // Map integer types using Clang's type size and signedness queries.
    // Note: C integer promotion is intentionally NOT preserved.  The
    // lowerExpr() visitor calls IgnoreParenImpCasts() on operands, which
    // strips implicit promotion casts.  Binary operators therefore use the
    // source-level (narrow) type rather than the C++ promoted type.  This
    // is a deliberate design choice: Arcanum verifies at the source type
    // level, producing stricter overflow checks (e.g., i8 bounds for
    // int8_t arithmetic rather than i32 bounds).  The spec mandates
    // rejecting non-fixed-width types and the tool operates on declared
    // types, not promoted types.
    if (canonical->isIntegerType()) {
      auto width = static_cast<unsigned>(astCtx.getTypeSize(canonical));
      bool isSigned = canonical->isSignedIntegerType();
      return arc::IntType::get(&mlirCtx, width, isSigned);
    }
    // Fallback for unrecognized types
    llvm::errs() << "warning: unrecognized type in getArcType, defaulting to "
                    "i32\n";
    DiagnosticTracker::recordFallback();
    return arc::IntType::get(&mlirCtx, 32, true);
  }

  /// Set overflow attribute on an arithmetic op.
  /// Unsigned types always get "wrap" (C++ unsigned wraps).
  /// Signed types use the function-level overflow mode.
  /// Per spec, default "trap" mode omits the attribute (SC-5).
  void setOverflowAttr(mlir::Operation* op) {
    auto resultType = op->getResult(0).getType();
    if (auto intType = llvm::dyn_cast<arc::IntType>(resultType)) {
      if (!intType.getIsSigned()) {
        op->setAttr("overflow", builder.getStringAttr("wrap"));
      } else if (currentOverflowMode != "trap") {
        // Only set overflow attribute on signed types when not "trap"
        // (default). Per spec: "Default: trap (attribute omitted)"
        op->setAttr("overflow", builder.getStringAttr(currentOverflowMode));
      }
      // When currentOverflowMode == "trap", omit attribute; getOverflowMode()
      // in WhyMLEmitter returns "trap" when no attribute is present.
    }
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

    // Get contract strings and overflow mode if present
    mlir::StringAttr requiresAttr;
    mlir::StringAttr ensuresAttr;
    currentOverflowMode = "trap"; // Reset to default for each function
    auto it = contracts.find(funcDecl);
    if (it != contracts.end()) {
      // Read overflow mode from contract
      currentOverflowMode = it->second.overflowMode;

      // Serialize contract expressions as string attributes.
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

    // Store overflow mode on the function.  Unlike arithmetic ops where
    // the default "trap" mode is represented by omitting the attribute
    // (see setOverflowAttr and SC-5), the FuncOp always carries the
    // overflow attribute explicitly -- including "trap" -- for
    // introspection and debugging purposes.
    funcOp->setAttr("overflow", builder.getStringAttr(currentOverflowMode));

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
      if (ret->getRetValue() != nullptr) {
        auto retVal = lowerExpr(ret->getRetValue(), valueMap);
        if (!retVal) {
          return; // Propagate failure; DiagnosticTracker already recorded it.
        }
        builder.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()), *retVal);
      } else {
        // Void return: create ReturnOp with no operand
        builder.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()),
                                      mlir::Value());
      }
    } else if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      for (const auto* d : declStmt->decls()) {
        if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(d)) {
          if (varDecl->hasInit()) {
            auto initVal = lowerExpr(varDecl->getInit(), valueMap);
            if (!initVal) {
              return; // Propagate failure.
            }
            auto loc = getLoc(varDecl->getLocation());
            auto varOp = builder.create<arc::VarOp>(
                loc, getArcType(varDecl->getType()), varDecl->getNameAsString(),
                *initVal);
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
      if (!cond) {
        return; // Propagate failure.
      }
      auto loc = getLoc(ifStmt->getIfLoc());
      auto ifOp = builder.create<arc::IfOp>(loc, mlir::TypeRange{}, *cond);

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
    } else if (const auto* exprStmt = llvm::dyn_cast<clang::Expr>(stmt)) {
      // Handle assignment expressions (e.g., x = expr)
      const auto* pureExpr = exprStmt->IgnoreParenImpCasts();
      if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(pureExpr)) {
        if (binOp->getOpcode() == clang::BO_Assign) {
          auto rhs = lowerExpr(binOp->getRHS(), valueMap);
          if (!rhs) {
            return; // Propagate failure.
          }
          auto loc = getLoc(binOp->getOperatorLoc());
          if (const auto* lhsRef = llvm::dyn_cast<clang::DeclRefExpr>(
                  binOp->getLHS()->IgnoreParenImpCasts())) {
            auto it = valueMap.find(lhsRef->getDecl());
            if (it != valueMap.end()) {
              builder.create<arc::AssignOp>(loc, it->second, *rhs);
              // Update valueMap so subsequent reads see the new value
              valueMap[lhsRef->getDecl()] = *rhs;
            }
          }
        }
      }
    }
  }

  std::optional<mlir::Value>
  lowerExpr(const clang::Expr* expr,
            llvm::DenseMap<const clang::ValueDecl*, mlir::Value>& valueMap) {
    expr = expr->IgnoreParenImpCasts();
    auto loc = getLoc(expr->getBeginLoc());

    if (const auto* intLit = llvm::dyn_cast<clang::IntegerLiteral>(expr)) {
      auto litType = getArcType(intLit->getType());
      auto intType = llvm::dyn_cast<arc::IntType>(litType);
      unsigned width = intType ? intType.getWidth() : 32;
      if (!intType) {
        // Unexpected type mapping failure; diagnostic for debugging.
        llvm::errs() << "warning: integer literal type did not map to "
                        "arc::IntType; defaulting to width 32\n";
      }
      // Use the APInt directly to avoid sign-extension issues for unsigned
      // 64-bit literals that exceed INT64_MAX (F4).
      llvm::APInt apVal = intLit->getValue();
      if (apVal.getBitWidth() != width) {
        apVal = apVal.sextOrTrunc(width);
      }
      return builder
          .create<arc::ConstantOp>(
              loc, litType,
              builder.getIntegerAttr(builder.getIntegerType(width), apVal))
          .getResult();
    }

    if (const auto* boolLit = llvm::dyn_cast<clang::CXXBoolLiteralExpr>(expr)) {
      return builder
          .create<arc::ConstantOp>(loc, arc::BoolType::get(&mlirCtx),
                                   builder.getBoolAttr(boolLit->getValue()))
          .getResult();
    }

    if (const auto* declRef = llvm::dyn_cast<clang::DeclRefExpr>(expr)) {
      auto it = valueMap.find(declRef->getDecl());
      if (it != valueMap.end()) {
        return it->second;
      }
      llvm::errs() << "warning: unknown declaration reference '"
                   << declRef->getDecl()->getNameAsString()
                   << "', lowering failed\n";
      DiagnosticTracker::recordFallback();
      return std::nullopt;
    }

    if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(expr)) {
      auto lhs = lowerExpr(binOp->getLHS(), valueMap);
      auto rhs = lowerExpr(binOp->getRHS(), valueMap);
      if (!lhs || !rhs) {
        return std::nullopt;
      }

      switch (binOp->getOpcode()) {
      case clang::BO_Add: {
        auto op = builder.create<arc::AddOp>(loc, lhs->getType(), *lhs, *rhs);
        setOverflowAttr(op);
        return op.getResult();
      }
      case clang::BO_Sub: {
        auto op = builder.create<arc::SubOp>(loc, lhs->getType(), *lhs, *rhs);
        setOverflowAttr(op);
        return op.getResult();
      }
      case clang::BO_Mul: {
        auto op = builder.create<arc::MulOp>(loc, lhs->getType(), *lhs, *rhs);
        setOverflowAttr(op);
        return op.getResult();
      }
      case clang::BO_Div: {
        auto op = builder.create<arc::DivOp>(loc, lhs->getType(), *lhs, *rhs);
        setOverflowAttr(op);
        return op.getResult();
      }
      case clang::BO_Rem: {
        auto op = builder.create<arc::RemOp>(loc, lhs->getType(), *lhs, *rhs);
        setOverflowAttr(op);
        return op.getResult();
      }
      case clang::BO_LT:
        return builder
            .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                builder.getStringAttr("lt"), *lhs, *rhs)
            .getResult();
      case clang::BO_LE:
        return builder
            .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                builder.getStringAttr("le"), *lhs, *rhs)
            .getResult();
      case clang::BO_GT:
        return builder
            .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                builder.getStringAttr("gt"), *lhs, *rhs)
            .getResult();
      case clang::BO_GE:
        return builder
            .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                builder.getStringAttr("ge"), *lhs, *rhs)
            .getResult();
      case clang::BO_EQ:
        return builder
            .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                builder.getStringAttr("eq"), *lhs, *rhs)
            .getResult();
      case clang::BO_NE:
        return builder
            .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                                builder.getStringAttr("ne"), *lhs, *rhs)
            .getResult();
      case clang::BO_LAnd:
        return builder
            .create<arc::AndOp>(loc, arc::BoolType::get(&mlirCtx), *lhs, *rhs)
            .getResult();
      case clang::BO_LOr:
        return builder
            .create<arc::OrOp>(loc, arc::BoolType::get(&mlirCtx), *lhs, *rhs)
            .getResult();
      default:
        llvm::errs() << "warning: unhandled binary operator opcode "
                     << binOp->getOpcodeStr() << ", lowering failed\n";
        DiagnosticTracker::recordFallback();
        return std::nullopt;
      }
    }

    if (const auto* unaryOp = llvm::dyn_cast<clang::UnaryOperator>(expr)) {
      auto operand = lowerExpr(unaryOp->getSubExpr(), valueMap);
      if (!operand) {
        return std::nullopt;
      }
      switch (unaryOp->getOpcode()) {
      case clang::UO_LNot:
        return builder
            .create<arc::NotOp>(loc, arc::BoolType::get(&mlirCtx), *operand)
            .getResult();
      case clang::UO_Minus: {
        auto operandIntType = llvm::dyn_cast<arc::IntType>(operand->getType());
        unsigned negWidth = operandIntType ? operandIntType.getWidth() : 32;
        auto zero = builder.create<arc::ConstantOp>(
            loc, operand->getType(),
            builder.getIntegerAttr(builder.getIntegerType(negWidth), 0));
        auto subOp =
            builder.create<arc::SubOp>(loc, operand->getType(), zero, *operand);
        setOverflowAttr(subOp);
        return subOp.getResult();
      }
      default:
        break;
      }
    }

    if (const auto* castExpr = llvm::dyn_cast<clang::CXXStaticCastExpr>(expr)) {
      auto subExpr = lowerExpr(castExpr->getSubExpr(), valueMap);
      if (!subExpr) {
        return std::nullopt;
      }
      auto targetType = getArcType(castExpr->getType());
      auto castOp = builder.create<arc::CastOp>(loc, targetType, *subExpr);
      // Propagate overflow mode onto CastOp (SC-4).
      // Casts inherit overflow mode from context (function-level).
      if (currentOverflowMode != "trap") {
        castOp->setAttr("overflow", builder.getStringAttr(currentOverflowMode));
      }
      return castOp.getResult();
    }

    // Propagate failure instead of polluting the MLIR module with zero
    // constants.  DiagnosticTracker records the count for the error message.
    llvm::errs() << "warning: unrecognized expression in lowering\n";
    DiagnosticTracker::recordFallback();
    return std::nullopt;
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
      const auto* op = BINARY_OP_STRINGS[static_cast<size_t>(expr->binaryOp)];
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
  std::string currentOverflowMode = "trap";
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context, clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts) {
  ArcLowering lowering(context, astContext, contracts);
  return lowering.lower();
}

} // namespace arcanum
