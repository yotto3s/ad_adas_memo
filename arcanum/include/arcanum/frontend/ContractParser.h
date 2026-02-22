#ifndef ARCANUM_FRONTEND_CONTRACTPARSER_H
#define ARCANUM_FRONTEND_CONTRACTPARSER_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace arcanum {

/// Simple expression AST node for contract expressions.
enum class ContractExprKind {
  IntLiteral,  // 42
  BoolLiteral, // true, false
  ParamRef,    // parameter name
  ResultRef,   // \result
  BinaryOp,    // a + b, a && b, a < b, etc.
  UnaryOp,     // !a, -a
};

struct ContractExpr;
using ContractExprPtr = std::shared_ptr<ContractExpr>;

enum class BinaryOpKind {
  Add,
  Sub,
  Mul,
  Div,
  Rem,
  Lt,
  Le,
  Gt,
  Ge,
  Eq,
  Ne,
  And,
  Or,
};

enum class UnaryOpKind {
  Not,
  Neg,
};

struct ContractExpr {
  ContractExprKind kind;

  // For IntLiteral
  int64_t intValue = 0;

  // For BoolLiteral
  bool boolValue = false;

  // For ParamRef
  std::string paramName;

  // For BinaryOp
  BinaryOpKind binaryOp = BinaryOpKind::Add;
  ContractExprPtr left;
  ContractExprPtr right;

  // For UnaryOp
  UnaryOpKind unaryOp = UnaryOpKind::Not;
  ContractExprPtr operand;

  static ContractExprPtr makeIntLiteral(int64_t val);
  static ContractExprPtr makeBoolLiteral(bool val);
  static ContractExprPtr makeParamRef(const std::string& name);
  static ContractExprPtr makeResultRef();
  static ContractExprPtr makeBinaryOp(BinaryOpKind op, ContractExprPtr lhs,
                                      ContractExprPtr rhs);
  static ContractExprPtr makeUnaryOp(UnaryOpKind op, ContractExprPtr operand);
};

struct ContractInfo {
  std::vector<ContractExprPtr> preconditions;
  std::vector<ContractExprPtr> postconditions;
};

/// Parse //@ requires: and //@ ensures: annotations from raw comments,
/// associating them with the FunctionDecl they immediately precede.
std::map<const clang::FunctionDecl*, ContractInfo>
parseContracts(clang::ASTContext& context);

} // namespace arcanum

#endif // ARCANUM_FRONTEND_CONTRACTPARSER_H
