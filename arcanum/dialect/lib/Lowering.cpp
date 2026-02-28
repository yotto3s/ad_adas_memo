#include "arcanum/dialect/Lowering.h"
#include "arcanum/DiagnosticTracker.h"
#include "arcanum/dialect/ArcDialect.h"
#include "arcanum/dialect/ArcOps.h"
#include "arcanum/dialect/ArcTypes.h"
#include "arcanum/frontend/ContractParser.h"

#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/RawCommentList.h"
#include "clang/AST/Stmt.h"
#include "clang/Basic/SourceManager.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <array>
#include <optional>
#include <string>

namespace arcanum {

/// Maximum line distance between a loop annotation comment and the loop
/// statement it annotates.
static constexpr unsigned MAX_ANNOTATION_DISTANCE = 10;

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

using ValueMap = llvm::DenseMap<const clang::ValueDecl*, mlir::Value>;

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

  // Finding 2: Extract contract expression joining into a helper.
  static std::string joinContractExprs(
      const std::vector<ContractExprPtr>& exprs,
      const std::function<std::string(const ContractExprPtr&)>& serialize) {
    std::string result;
    for (size_t i = 0; i < exprs.size(); ++i) {
      if (i > 0) {
        result += " && ";
      }
      result += serialize(exprs[i]);
    }
    return result;
  }

  // Finding 1a: Build the MLIR function type from a FunctionDecl.
  mlir::FunctionType buildFuncType(const clang::FunctionDecl* funcDecl) {
    llvm::SmallVector<mlir::Type> paramTypes;
    for (const auto* param : funcDecl->parameters()) {
      paramTypes.push_back(getArcType(param->getType()));
    }
    mlir::Type resultType = getArcType(funcDecl->getReturnType());
    return builder.getFunctionType(paramTypes, {resultType});
  }

  // Finding 1b: Build requires/ensures string attrs and set overflow mode.
  void buildContractAttrs(const clang::FunctionDecl* funcDecl,
                          mlir::StringAttr& requiresAttr,
                          mlir::StringAttr& ensuresAttr) {
    currentOverflowMode = "trap"; // Reset to default for each function
    auto it = contracts.find(funcDecl);
    if (it == contracts.end()) {
      return;
    }

    currentOverflowMode = it->second.overflowMode;

    auto serialize = [this](const ContractExprPtr& e) {
      return serializeExpr(e);
    };
    std::string reqStr = joinContractExprs(it->second.preconditions, serialize);
    std::string ensStr =
        joinContractExprs(it->second.postconditions, serialize);

    if (!reqStr.empty()) {
      requiresAttr = builder.getStringAttr(reqStr);
    }
    if (!ensStr.empty()) {
      ensuresAttr = builder.getStringAttr(ensStr);
    }
  }

  // Finding 1c: Create a FuncOp and attach contract + overflow + param attrs.
  arc::FuncOp createFuncOpWithAttrs(mlir::Location loc, llvm::StringRef name,
                                    mlir::FunctionType funcType,
                                    mlir::StringAttr requiresAttr,
                                    mlir::StringAttr ensuresAttr,
                                    const clang::FunctionDecl* funcDecl) {
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

    return funcOp;
  }

  // Finding 1d: Emplace entry block and add one argument per param type.
  mlir::Block& buildEntryBlock(arc::FuncOp funcOp, mlir::FunctionType funcType,
                               mlir::Location loc) {
    auto& entryBlock = funcOp.getBody().emplaceBlock();
    for (auto paramType : funcType.getInputs()) {
      entryBlock.addArgument(paramType, loc);
    }
    return entryBlock;
  }

  // Finding 1e: Populate valueMap from FunctionDecl params to block args.
  ValueMap buildParamValueMap(const clang::FunctionDecl* funcDecl,
                              mlir::Block& entryBlock) {
    ValueMap valueMap;
    for (size_t i = 0; i < funcDecl->getNumParams(); ++i) {
      valueMap[funcDecl->getParamDecl(i)] = entryBlock.getArgument(i);
    }
    return valueMap;
  }

  void lowerFunction(clang::FunctionDecl* funcDecl) {
    labelsSeen.clear(); // Reset per-function label tracking (SC-2)
    auto loc = getLoc(funcDecl->getLocation());
    auto name = funcDecl->getNameAsString();

    auto funcType = buildFuncType(funcDecl);

    mlir::StringAttr requiresAttr;
    mlir::StringAttr ensuresAttr;
    buildContractAttrs(funcDecl, requiresAttr, ensuresAttr);

    auto funcOp = createFuncOpWithAttrs(loc, name, funcType, requiresAttr,
                                        ensuresAttr, funcDecl);

    auto& entryBlock = buildEntryBlock(funcOp, funcType, loc);
    auto valueMap = buildParamValueMap(funcDecl, entryBlock);

    // Lower function body
    auto savedIp = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&entryBlock);
    lowerStmt(funcDecl->getBody(), valueMap);
    builder.restoreInsertionPoint(savedIp);
  }

  // Finding 3a: Lower a return statement.
  void lowerReturnStmt(const clang::ReturnStmt* ret, ValueMap& valueMap) {
    if (ret->getRetValue() != nullptr) {
      auto retVal = lowerExpr(ret->getRetValue(), valueMap);
      if (!retVal) {
        return; // Propagate failure; DiagnosticTracker already recorded it.
      }
      builder.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()), *retVal);
    } else {
      // Void return: create ReturnOp with no operand
      builder.create<arc::ReturnOp>(getLoc(ret->getReturnLoc()), mlir::Value());
    }
  }

  // Finding 3b: Lower a variable declaration statement.
  void lowerDeclStmt(const clang::DeclStmt* declStmt, ValueMap& valueMap) {
    for (const auto* d : declStmt->decls()) {
      if (const auto* varDecl = llvm::dyn_cast<clang::VarDecl>(d)) {
        if (varDecl->hasInit()) {
          auto initVal = lowerExpr(varDecl->getInit(), valueMap);
          if (!initVal) {
            return; // Propagate failure.
          }
          auto loc = getLoc(varDecl->getLocation());
          auto varOp =
              builder.create<arc::VarOp>(loc, getArcType(varDecl->getType()),
                                         varDecl->getNameAsString(), *initVal);
          valueMap[varDecl] = varOp.getResult();
        }
      }
    }
  }

  // Finding 4: Lower a single Clang Stmt into a new block inside `region`.
  void lowerStmtIntoRegion(mlir::Region& region, const clang::Stmt* stmt,
                           ValueMap& valueMap) {
    auto& block = region.emplaceBlock();
    auto savedIp = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&block);
    lowerStmt(stmt, valueMap);
    builder.restoreInsertionPoint(savedIp);
  }

  // Finding 3c: Lower an if statement.
  void lowerIfStmt(const clang::IfStmt* ifStmt, ValueMap& valueMap) {
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

    lowerStmtIntoRegion(ifOp.getThenRegion(), ifStmt->getThen(), valueMap);

    if (ifStmt->getElse() != nullptr) {
      lowerStmtIntoRegion(ifOp.getElseRegion(), ifStmt->getElse(), valueMap);
    }
  }

  // --- Loop lowering helpers (Slice 3) ---

  /// Collect loop annotation lines from comments preceding a Stmt location.
  /// Scans all comments in the same file and selects those ending within 10
  /// lines before the statement.
  ///
  /// Known limitation (CR-4): distance-based collection does not filter
  /// out comments that belong to an intervening loop. If two loops are
  /// within MAX_ANNOTATION_DISTANCE lines of each other, annotations for
  /// the first loop may be incorrectly attributed to the second. In
  /// practice, loop annotations immediately precede their loop statement.
  LoopContractInfo collectLoopAnnotations(clang::SourceLocation stmtLoc) {
    LoopContractInfo loopInfo;
    if (!stmtLoc.isValid()) {
      return loopInfo;
    }
    auto& sm = astCtx.getSourceManager();
    auto fileId = sm.getFileID(stmtLoc);
    auto* commentsMap = astCtx.Comments.getCommentsInFile(fileId);
    if (commentsMap == nullptr) {
      return loopInfo;
    }
    unsigned stmtBeginLine = sm.getPresumedLineNumber(stmtLoc);
    for (const auto& [offset, comment] : *commentsMap) {
      auto commentEnd = comment->getEndLoc();
      if (!commentEnd.isValid()) {
        continue;
      }
      if (!sm.isBeforeInTranslationUnit(commentEnd, stmtLoc)) {
        continue;
      }
      unsigned commentEndLine = sm.getPresumedLineNumber(commentEnd);
      if (stmtBeginLine - commentEndLine > MAX_ANNOTATION_DISTANCE) {
        continue;
      }
      auto rawText = comment->getRawText(sm);
      auto annotationLines = extractAnnotationLines(rawText);
      for (const auto& line : annotationLines) {
        llvm::StringRef lineRef(line);
        if (lineRef.starts_with("loop_") || lineRef.starts_with("label:")) {
          applyLoopAnnotationLine(lineRef, loopInfo);
        }
      }
    }
    return loopInfo;
  }

  /// Attach loop contract attributes (invariant, variant, assigns, label)
  /// from a LoopContractInfo onto an arc.loop operation.
  void attachLoopContractAttrs(arc::LoopOp loopOp,
                               const LoopContractInfo& loopInfo) {
    if (!loopInfo.invariant.empty()) {
      loopOp->setAttr("invariant", builder.getStringAttr(loopInfo.invariant));
    }
    if (!loopInfo.variant.empty()) {
      loopOp->setAttr("variant", builder.getStringAttr(loopInfo.variant));
    }
    if (!loopInfo.assigns.empty()) {
      // NOTE (CQ-3): Duplicates joinNames in Passes.cpp.  Consolidation
      // deferred pending a shared utility.
      std::string assignsStr;
      for (size_t i = 0; i < loopInfo.assigns.size(); ++i) {
        if (i > 0) {
          assignsStr += ", ";
        }
        assignsStr += loopInfo.assigns[i];
      }
      loopOp->setAttr("assigns", builder.getStringAttr(assignsStr));
    }
    if (!loopInfo.label.empty()) {
      if (!labelsSeen.insert(loopInfo.label).second) {
        loopOp.emitError("duplicate loop label '")
            << loopInfo.label << "' within function";
      }
      loopOp->setAttr("label", builder.getStringAttr(loopInfo.label));
    }
  }

  /// Lower a condition expression into a region, terminating with
  /// arc.condition.
  void lowerCondIntoRegion(mlir::Region& region, const clang::Expr* condExpr,
                           ValueMap& valueMap) {
    auto& block = region.emplaceBlock();
    auto savedIp = builder.saveInsertionPoint();
    builder.setInsertionPointToEnd(&block);
    auto condVal = lowerExpr(condExpr, valueMap);
    if (condVal) {
      builder.create<arc::ConditionOp>(getLoc(condExpr->getBeginLoc()),
                                       *condVal);
    }
    builder.restoreInsertionPoint(savedIp);
  }

  /// Append an arc.yield terminator to the last block in a region, if the
  /// region is non-empty and the last operation is not already a terminator.
  void appendYieldTerminator(mlir::Region& region) {
    if (region.empty()) {
      return;
    }
    auto& block = region.back();
    if (block.empty() ||
        !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
      auto savedIp = builder.saveInsertionPoint();
      builder.setInsertionPointToEnd(&block);
      builder.create<arc::YieldOp>(builder.getUnknownLoc());
      builder.restoreInsertionPoint(savedIp);
    }
  }

  void lowerForStmt(const clang::ForStmt* forStmt, ValueMap& valueMap) {
    auto loc = getLoc(forStmt->getForLoc());
    auto loopInfo = collectLoopAnnotations(forStmt->getForLoc());

    auto loopOp = builder.create<arc::LoopOp>(loc);
    loopOp->setAttr("condition_first", builder.getBoolAttr(true));
    attachLoopContractAttrs(loopOp, loopInfo);

    if (forStmt->getInit()) {
      lowerStmtIntoRegion(loopOp.getInitRegion(), forStmt->getInit(), valueMap);
      appendYieldTerminator(loopOp.getInitRegion());
    }
    if (forStmt->getCond()) {
      lowerCondIntoRegion(loopOp.getCondRegion(), forStmt->getCond(), valueMap);
    }
    if (forStmt->getInc()) {
      lowerStmtIntoRegion(loopOp.getUpdateRegion(), forStmt->getInc(),
                          valueMap);
      appendYieldTerminator(loopOp.getUpdateRegion());
    }
    if (forStmt->getBody()) {
      lowerStmtIntoRegion(loopOp.getBodyRegion(), forStmt->getBody(), valueMap);
      appendYieldTerminator(loopOp.getBodyRegion());
    }
  }

  void lowerWhileStmt(const clang::WhileStmt* whileStmt, ValueMap& valueMap) {
    auto loc = getLoc(whileStmt->getWhileLoc());
    auto loopInfo = collectLoopAnnotations(whileStmt->getWhileLoc());

    auto loopOp = builder.create<arc::LoopOp>(loc);
    loopOp->setAttr("condition_first", builder.getBoolAttr(true));
    attachLoopContractAttrs(loopOp, loopInfo);

    lowerCondIntoRegion(loopOp.getCondRegion(), whileStmt->getCond(), valueMap);

    if (whileStmt->getBody()) {
      lowerStmtIntoRegion(loopOp.getBodyRegion(), whileStmt->getBody(),
                          valueMap);
      appendYieldTerminator(loopOp.getBodyRegion());
    }
  }

  void lowerDoStmt(const clang::DoStmt* doStmt, ValueMap& valueMap) {
    auto loc = getLoc(doStmt->getDoLoc());
    auto loopInfo = collectLoopAnnotations(doStmt->getDoLoc());

    auto loopOp = builder.create<arc::LoopOp>(loc);
    loopOp->setAttr("condition_first", builder.getBoolAttr(false));
    attachLoopContractAttrs(loopOp, loopInfo);

    lowerCondIntoRegion(loopOp.getCondRegion(), doStmt->getCond(), valueMap);

    if (doStmt->getBody()) {
      lowerStmtIntoRegion(loopOp.getBodyRegion(), doStmt->getBody(), valueMap);
      appendYieldTerminator(loopOp.getBodyRegion());
    }
  }

  // Finding 3d: Lower an assignment expression statement.
  void lowerAssignmentExpr(const clang::BinaryOperator* binOp,
                           ValueMap& valueMap) {
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

  void lowerStmt(const clang::Stmt* stmt, ValueMap& valueMap) {
    if (const auto* compound = llvm::dyn_cast<clang::CompoundStmt>(stmt)) {
      for (const auto* child : compound->body()) {
        lowerStmt(child, valueMap);
      }
    } else if (const auto* ret = llvm::dyn_cast<clang::ReturnStmt>(stmt)) {
      lowerReturnStmt(ret, valueMap);
    } else if (const auto* declStmt = llvm::dyn_cast<clang::DeclStmt>(stmt)) {
      lowerDeclStmt(declStmt, valueMap);
    } else if (const auto* ifStmt = llvm::dyn_cast<clang::IfStmt>(stmt)) {
      lowerIfStmt(ifStmt, valueMap);
    } else if (const auto* forStmt = llvm::dyn_cast<clang::ForStmt>(stmt)) {
      lowerForStmt(forStmt, valueMap);
    } else if (const auto* whileStmt = llvm::dyn_cast<clang::WhileStmt>(stmt)) {
      lowerWhileStmt(whileStmt, valueMap);
    } else if (const auto* doStmt = llvm::dyn_cast<clang::DoStmt>(stmt)) {
      lowerDoStmt(doStmt, valueMap);
    } else if (llvm::isa<clang::BreakStmt>(stmt)) {
      builder.create<arc::BreakOp>(getLoc(stmt->getBeginLoc()));
    } else if (llvm::isa<clang::ContinueStmt>(stmt)) {
      builder.create<arc::ContinueOp>(getLoc(stmt->getBeginLoc()));
    } else if (const auto* exprStmt = llvm::dyn_cast<clang::Expr>(stmt)) {
      // Handle assignment expressions (e.g., x = expr)
      const auto* pureExpr = exprStmt->IgnoreParenImpCasts();
      if (const auto* binOp = llvm::dyn_cast<clang::BinaryOperator>(pureExpr)) {
        if (binOp->getOpcode() == clang::BO_Assign) {
          lowerAssignmentExpr(binOp, valueMap);
        }
      }
    }
  }

  // Finding 5a: Lower arithmetic binary ops (with overflow attribute).
  mlir::Value lowerArithBinOp(clang::BinaryOperatorKind opcode,
                              mlir::Location loc, mlir::Value lhs,
                              mlir::Value rhs) {
    mlir::Operation* op = nullptr;
    switch (opcode) {
    case clang::BO_Add:
      op = builder.create<arc::AddOp>(loc, lhs.getType(), lhs, rhs);
      break;
    case clang::BO_Sub:
      op = builder.create<arc::SubOp>(loc, lhs.getType(), lhs, rhs);
      break;
    case clang::BO_Mul:
      op = builder.create<arc::MulOp>(loc, lhs.getType(), lhs, rhs);
      break;
    case clang::BO_Div:
      op = builder.create<arc::DivOp>(loc, lhs.getType(), lhs, rhs);
      break;
    case clang::BO_Rem:
      op = builder.create<arc::RemOp>(loc, lhs.getType(), lhs, rhs);
      break;
    default:
      llvm_unreachable("lowerArithBinOp called with non-arithmetic opcode");
    }
    setOverflowAttr(op);
    return op->getResult(0);
  }

  // Finding 5b: Lower comparison binary ops (no overflow attribute).
  mlir::Value lowerCmpBinOp(llvm::StringRef predicate, mlir::Location loc,
                            mlir::Value lhs, mlir::Value rhs) {
    return builder
        .create<arc::CmpOp>(loc, arc::BoolType::get(&mlirCtx),
                            builder.getStringAttr(predicate), lhs, rhs)
        .getResult();
  }

  std::optional<mlir::Value> lowerExpr(const clang::Expr* expr,
                                       ValueMap& valueMap) {
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
      case clang::BO_Add:
      case clang::BO_Sub:
      case clang::BO_Mul:
      case clang::BO_Div:
      case clang::BO_Rem:
        return lowerArithBinOp(binOp->getOpcode(), loc, *lhs, *rhs);
      case clang::BO_LT:
        return lowerCmpBinOp("lt", loc, *lhs, *rhs);
      case clang::BO_LE:
        return lowerCmpBinOp("le", loc, *lhs, *rhs);
      case clang::BO_GT:
        return lowerCmpBinOp("gt", loc, *lhs, *rhs);
      case clang::BO_GE:
        return lowerCmpBinOp("ge", loc, *lhs, *rhs);
      case clang::BO_EQ:
        return lowerCmpBinOp("eq", loc, *lhs, *rhs);
      case clang::BO_NE:
        return lowerCmpBinOp("ne", loc, *lhs, *rhs);
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
      case clang::UO_Minus:
        return lowerUnaryNegation(loc, *operand);
      default:
        break;
      }
    }

    if (const auto* castExpr = llvm::dyn_cast<clang::CXXStaticCastExpr>(expr)) {
      return lowerStaticCast(castExpr, loc, valueMap);
    }

    // Propagate failure instead of polluting the MLIR module with zero
    // constants.  DiagnosticTracker records the count for the error message.
    llvm::errs() << "warning: unrecognized expression in lowering\n";
    DiagnosticTracker::recordFallback();
    return std::nullopt;
  }

  // Finding 6: Extract unary negation (zero-subtraction pattern).
  mlir::Value lowerUnaryNegation(mlir::Location loc, mlir::Value operand) {
    auto operandIntType = llvm::dyn_cast<arc::IntType>(operand.getType());
    unsigned negWidth = operandIntType ? operandIntType.getWidth() : 32;
    auto zero = builder.create<arc::ConstantOp>(
        loc, operand.getType(),
        builder.getIntegerAttr(builder.getIntegerType(negWidth), 0));
    auto subOp =
        builder.create<arc::SubOp>(loc, operand.getType(), zero, operand);
    setOverflowAttr(subOp);
    return subOp.getResult();
  }

  // Finding 7: Extract static cast emission with overflow propagation.
  std::optional<mlir::Value>
  lowerStaticCast(const clang::CXXStaticCastExpr* castExpr, mlir::Location loc,
                  ValueMap& valueMap) {
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
  /// Track label names within the current function for uniqueness validation
  /// (SC-2: spec requires label uniqueness within a function).
  llvm::StringSet<> labelsSeen;
};

} // namespace

mlir::OwningOpRef<mlir::ModuleOp> lowerToArc(
    mlir::MLIRContext& context, clang::ASTContext& astContext,
    const std::map<const clang::FunctionDecl*, ContractInfo>& contracts) {
  ArcLowering lowering(context, astContext, contracts);
  return lowering.lower();
}

} // namespace arcanum
