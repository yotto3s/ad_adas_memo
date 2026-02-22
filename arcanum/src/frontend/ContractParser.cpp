#include "ContractParser.h"

#include "clang/AST/ASTContext.h"
#include "clang/AST/Comment.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclGroup.h"
#include "clang/Basic/SourceManager.h"

#include "llvm/ADT/StringRef.h"

#include <cassert>
#include <cctype>
#include <optional>
#include <sstream>
#include <string>

namespace arcanum {

// ============================================================
// ContractExpr factory functions
// ============================================================

ContractExprPtr makeIntLiteral(int64_t value) {
    auto expr = std::make_unique<ContractExpr>();
    expr->node = IntLiteral{value};
    return expr;
}

ContractExprPtr makeBoolLiteral(bool value) {
    auto expr = std::make_unique<ContractExpr>();
    expr->node = BoolLiteral{value};
    return expr;
}

ContractExprPtr makeIdentifier(std::string name) {
    auto expr = std::make_unique<ContractExpr>();
    expr->node = Identifier{std::move(name)};
    return expr;
}

ContractExprPtr makeResultRef() {
    auto expr = std::make_unique<ContractExpr>();
    expr->node = ResultRef{};
    return expr;
}

ContractExprPtr makeBinaryExpr(BinOp op, ContractExprPtr lhs,
                               ContractExprPtr rhs) {
    auto expr = std::make_unique<ContractExpr>();
    expr->node = BinaryExpr{op, std::move(lhs), std::move(rhs)};
    return expr;
}

ContractExprPtr makeUnaryExpr(UnaryOp op, ContractExprPtr operand) {
    auto expr = std::make_unique<ContractExpr>();
    expr->node = UnaryExpr{op, std::move(operand)};
    return expr;
}

// ============================================================
// ContractExpr pretty printer
// ============================================================

namespace {

std::string binOpToString(BinOp op) {
    switch (op) {
    case BinOp::Add: return "+";
    case BinOp::Sub: return "-";
    case BinOp::Mul: return "*";
    case BinOp::Div: return "/";
    case BinOp::Mod: return "%";
    case BinOp::Lt:  return "<";
    case BinOp::Le:  return "<=";
    case BinOp::Gt:  return ">";
    case BinOp::Ge:  return ">=";
    case BinOp::Eq:  return "==";
    case BinOp::Ne:  return "!=";
    case BinOp::And: return "&&";
    case BinOp::Or:  return "||";
    }
    return "?";
}

} // anonymous namespace

std::string contractExprToString(const ContractExpr &expr) {
    return std::visit(
        [](const auto &node) -> std::string {
            using T = std::decay_t<decltype(node)>;

            if constexpr (std::is_same_v<T, IntLiteral>) {
                return std::to_string(node.value);
            } else if constexpr (std::is_same_v<T, BoolLiteral>) {
                return node.value ? "true" : "false";
            } else if constexpr (std::is_same_v<T, Identifier>) {
                return node.name;
            } else if constexpr (std::is_same_v<T, ResultRef>) {
                return "\\result";
            } else if constexpr (std::is_same_v<T, BinaryExpr>) {
                return "(" + contractExprToString(*node.lhs) + " " +
                       binOpToString(node.op) + " " +
                       contractExprToString(*node.rhs) + ")";
            } else if constexpr (std::is_same_v<T, UnaryExpr>) {
                std::string opStr =
                    node.op == UnaryOp::Not ? "!" : "-";
                return opStr + contractExprToString(*node.operand);
            }
        },
        expr.node);
}

// ============================================================
// Expression Lexer
// ============================================================

namespace {

enum class TokenKind {
    // Literals and identifiers
    IntLit,
    Ident,
    ResultKw,   // \result
    TrueKw,     // true
    FalseKw,    // false

    // Operators
    Plus,       // +
    Minus,      // -
    Star,       // *
    Slash,      // /
    Percent,    // %
    Lt,         // <
    Le,         // <=
    Gt,         // >
    Ge,         // >=
    EqEq,       // ==
    BangEq,     // !=
    AmpAmp,     // &&
    PipePipe,   // ||
    Bang,       // !

    // Grouping
    LParen,     // (
    RParen,     // )

    // End of input
    Eof,

    // Error
    Error,
};

struct Token {
    TokenKind kind;
    std::string text;
    unsigned column;
};

class Lexer {
public:
    explicit Lexer(const std::string &source) : source_(source), pos_(0) {}

    Token next() {
        skipWhitespace();

        if (pos_ >= source_.size())
            return Token{TokenKind::Eof, "", static_cast<unsigned>(pos_)};

        unsigned startCol = static_cast<unsigned>(pos_);
        char c = source_[pos_];

        // Integer literal
        if (std::isdigit(c)) {
            return lexNumber(startCol);
        }

        // Identifier or keyword
        if (std::isalpha(c) || c == '_') {
            return lexIdentifier(startCol);
        }

        // \result
        if (c == '\\') {
            return lexBackslash(startCol);
        }

        // Two-character operators
        if (pos_ + 1 < source_.size()) {
            char next = source_[pos_ + 1];
            if (c == '<' && next == '=') {
                pos_ += 2;
                return Token{TokenKind::Le, "<=", startCol};
            }
            if (c == '>' && next == '=') {
                pos_ += 2;
                return Token{TokenKind::Ge, ">=", startCol};
            }
            if (c == '=' && next == '=') {
                pos_ += 2;
                return Token{TokenKind::EqEq, "==", startCol};
            }
            if (c == '!' && next == '=') {
                pos_ += 2;
                return Token{TokenKind::BangEq, "!=", startCol};
            }
            if (c == '&' && next == '&') {
                pos_ += 2;
                return Token{TokenKind::AmpAmp, "&&", startCol};
            }
            if (c == '|' && next == '|') {
                pos_ += 2;
                return Token{TokenKind::PipePipe, "||", startCol};
            }
        }

        // Single-character operators
        pos_++;
        switch (c) {
        case '+': return Token{TokenKind::Plus, "+", startCol};
        case '-': return Token{TokenKind::Minus, "-", startCol};
        case '*': return Token{TokenKind::Star, "*", startCol};
        case '/': return Token{TokenKind::Slash, "/", startCol};
        case '%': return Token{TokenKind::Percent, "%", startCol};
        case '<': return Token{TokenKind::Lt, "<", startCol};
        case '>': return Token{TokenKind::Gt, ">", startCol};
        case '!': return Token{TokenKind::Bang, "!", startCol};
        case '(': return Token{TokenKind::LParen, "(", startCol};
        case ')': return Token{TokenKind::RParen, ")", startCol};
        default:
            return Token{TokenKind::Error,
                         std::string(1, c), startCol};
        }
    }

private:
    void skipWhitespace() {
        while (pos_ < source_.size() && std::isspace(source_[pos_]))
            pos_++;
    }

    Token lexNumber(unsigned startCol) {
        std::string text;
        while (pos_ < source_.size() && std::isdigit(source_[pos_])) {
            text += source_[pos_];
            pos_++;
        }
        return Token{TokenKind::IntLit, text, startCol};
    }

    Token lexIdentifier(unsigned startCol) {
        std::string text;
        while (pos_ < source_.size() &&
               (std::isalnum(source_[pos_]) || source_[pos_] == '_')) {
            text += source_[pos_];
            pos_++;
        }

        if (text == "true")
            return Token{TokenKind::TrueKw, text, startCol};
        if (text == "false")
            return Token{TokenKind::FalseKw, text, startCol};

        return Token{TokenKind::Ident, text, startCol};
    }

    Token lexBackslash(unsigned startCol) {
        // Expect \result
        pos_++; // skip '\'
        std::string text;
        while (pos_ < source_.size() &&
               (std::isalnum(source_[pos_]) || source_[pos_] == '_')) {
            text += source_[pos_];
            pos_++;
        }
        if (text == "result")
            return Token{TokenKind::ResultKw, "\\result", startCol};

        return Token{TokenKind::Error, "\\" + text, startCol};
    }

    const std::string &source_;
    std::size_t pos_;
};

// ============================================================
// Expression Parser (Recursive Descent)
// ============================================================

/// Precedence-climbing expression parser.
///
/// Precedence (lowest to highest):
///   1. || (logical or)
///   2. && (logical and)
///   3. ==, != (equality)
///   4. <, <=, >, >= (comparison)
///   5. +, - (additive)
///   6. *, /, % (multiplicative)
///   7. !, - (unary)
///   8. atoms: int, bool, ident, \result, (expr)
class ExprParser {
public:
    ExprParser(const std::string &source, unsigned sourceLine)
        : lexer_(source), sourceLine_(sourceLine) {
        advance();
    }

    ExprParseResult parse() {
        auto expr = parseOr();
        if (!errors_.empty())
            return ExprParseResult{nullptr, std::move(errors_)};
        if (current_.kind != TokenKind::Eof) {
            addError("unexpected token after expression: '" +
                     current_.text + "'");
            return ExprParseResult{nullptr, std::move(errors_)};
        }
        return ExprParseResult{std::move(expr), {}};
    }

private:
    // --- Precedence levels ---

    ContractExprPtr parseOr() {
        auto lhs = parseAnd();
        while (current_.kind == TokenKind::PipePipe) {
            advance();
            auto rhs = parseAnd();
            lhs = makeBinaryExpr(BinOp::Or, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ContractExprPtr parseAnd() {
        auto lhs = parseEquality();
        while (current_.kind == TokenKind::AmpAmp) {
            advance();
            auto rhs = parseEquality();
            lhs = makeBinaryExpr(BinOp::And, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ContractExprPtr parseEquality() {
        auto lhs = parseComparison();
        while (current_.kind == TokenKind::EqEq ||
               current_.kind == TokenKind::BangEq) {
            BinOp op = current_.kind == TokenKind::EqEq ? BinOp::Eq : BinOp::Ne;
            advance();
            auto rhs = parseComparison();
            lhs = makeBinaryExpr(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ContractExprPtr parseComparison() {
        auto lhs = parseAdditive();
        while (current_.kind == TokenKind::Lt ||
               current_.kind == TokenKind::Le ||
               current_.kind == TokenKind::Gt ||
               current_.kind == TokenKind::Ge) {
            BinOp op;
            switch (current_.kind) {
            case TokenKind::Lt: op = BinOp::Lt; break;
            case TokenKind::Le: op = BinOp::Le; break;
            case TokenKind::Gt: op = BinOp::Gt; break;
            case TokenKind::Ge: op = BinOp::Ge; break;
            default: __builtin_unreachable();
            }
            advance();
            auto rhs = parseAdditive();
            lhs = makeBinaryExpr(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ContractExprPtr parseAdditive() {
        auto lhs = parseMultiplicative();
        while (current_.kind == TokenKind::Plus ||
               current_.kind == TokenKind::Minus) {
            BinOp op =
                current_.kind == TokenKind::Plus ? BinOp::Add : BinOp::Sub;
            advance();
            auto rhs = parseMultiplicative();
            lhs = makeBinaryExpr(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ContractExprPtr parseMultiplicative() {
        auto lhs = parseUnary();
        while (current_.kind == TokenKind::Star ||
               current_.kind == TokenKind::Slash ||
               current_.kind == TokenKind::Percent) {
            BinOp op;
            switch (current_.kind) {
            case TokenKind::Star:    op = BinOp::Mul; break;
            case TokenKind::Slash:   op = BinOp::Div; break;
            case TokenKind::Percent: op = BinOp::Mod; break;
            default: __builtin_unreachable();
            }
            advance();
            auto rhs = parseUnary();
            lhs = makeBinaryExpr(op, std::move(lhs), std::move(rhs));
        }
        return lhs;
    }

    ContractExprPtr parseUnary() {
        if (current_.kind == TokenKind::Bang) {
            advance();
            auto operand = parseUnary();
            return makeUnaryExpr(UnaryOp::Not, std::move(operand));
        }
        if (current_.kind == TokenKind::Minus) {
            advance();
            auto operand = parseUnary();
            return makeUnaryExpr(UnaryOp::Neg, std::move(operand));
        }
        return parseAtom();
    }

    ContractExprPtr parseAtom() {
        switch (current_.kind) {
        case TokenKind::IntLit: {
            int64_t val = std::stoll(current_.text);
            advance();
            return makeIntLiteral(val);
        }
        case TokenKind::TrueKw:
            advance();
            return makeBoolLiteral(true);
        case TokenKind::FalseKw:
            advance();
            return makeBoolLiteral(false);
        case TokenKind::Ident: {
            std::string name = current_.text;
            advance();
            return makeIdentifier(std::move(name));
        }
        case TokenKind::ResultKw:
            advance();
            return makeResultRef();
        case TokenKind::LParen: {
            advance();
            auto inner = parseOr();
            if (current_.kind != TokenKind::RParen) {
                addError("expected ')' to close parenthesized expression");
                return inner;
            }
            advance();
            return inner;
        }
        case TokenKind::Eof:
            addError("unexpected end of expression");
            return makeIntLiteral(0);
        case TokenKind::Error:
            addError("unexpected character: '" + current_.text + "'");
            advance();
            return makeIntLiteral(0);
        default:
            addError("unexpected token: '" + current_.text + "'");
            advance();
            return makeIntLiteral(0);
        }
    }

    // --- Helpers ---

    void advance() { current_ = lexer_.next(); }

    void addError(std::string message) {
        errors_.push_back(ContractParseError{
            sourceLine_, current_.column, std::move(message)});
    }

    Lexer lexer_;
    Token current_;
    unsigned sourceLine_;
    std::vector<ContractParseError> errors_;
};

} // anonymous namespace

// ============================================================
// Public parseContractExpr
// ============================================================

ExprParseResult parseContractExpr(const std::string &source,
                                  unsigned sourceLine) {
    ExprParser parser(source, sourceLine);
    return parser.parse();
}

// ============================================================
// ContractParserResult
// ============================================================

const ContractInfo *
ContractParserResult::findContract(const clang::FunctionDecl *fn) const {
    for (const auto &[decl, info] : contracts) {
        if (decl == fn)
            return &info;
    }
    return nullptr;
}

// ============================================================
// ContractParser
// ============================================================

ContractParser::ContractParser(clang::ASTContext &context)
    : context_(context) {}

ContractParserResult ContractParser::parse() {
    ContractParserResult result;
    auto &sm = context_.getSourceManager();

    // Collect all comments from the raw comment list.
    auto &rawComments = context_.getRawCommentList();

    // Build a map of source line -> list of annotations for that line range.
    // Then associate each annotation group with its following function decl.

    // Iterate over all top-level decls to find functions.
    auto *tu = context_.getTranslationUnitDecl();
    for (auto *decl : tu->decls()) {
        auto *fn = llvm::dyn_cast<clang::FunctionDecl>(decl);
        if (!fn)
            continue;

        // Skip functions not in the main file.
        if (!sm.isInMainFile(fn->getLocation()))
            continue;

        // Get the raw comment associated with this function (Clang looks
        // at comments immediately preceding the declaration).
        const clang::RawComment *comment =
            context_.getRawCommentForDeclNoCache(fn);
        if (!comment)
            continue;

        std::string commentText = comment->getRawText(sm).str();

        // Split comment text into individual lines and parse each.
        ContractInfo info;
        std::istringstream stream(commentText);
        std::string line;
        unsigned lineNum =
            sm.getPresumedLineNumber(comment->getBeginLoc());

        while (std::getline(stream, line)) {
            // Strip leading whitespace.
            llvm::StringRef lineRef(line);
            lineRef = lineRef.ltrim();

            // Check for //@ prefix.
            if (!lineRef.starts_with("//@")) {
                lineNum++;
                continue;
            }

            // Strip the //@ prefix and leading whitespace.
            llvm::StringRef annotationText = lineRef.substr(3).ltrim();

            auto annotation =
                parseAnnotationLine(annotationText, lineNum, result.errors);
            if (annotation) {
                if (annotation->kind == ContractKind::Requires) {
                    info.preconditions.push_back(std::move(*annotation));
                } else {
                    info.postconditions.push_back(std::move(*annotation));
                }
            }
            lineNum++;
        }

        if (!info.empty()) {
            result.contracts.emplace_back(fn, std::move(info));
        }
    }

    return result;
}

std::string
ContractParser::extractCommentText(const clang::RawComment *comment) const {
    return comment->getRawText(context_.getSourceManager()).str();
}

bool ContractParser::isContractComment(llvm::StringRef text) const {
    return text.ltrim().starts_with("//@");
}

std::optional<ContractAnnotation> ContractParser::parseAnnotationLine(
    llvm::StringRef line, unsigned sourceLine,
    std::vector<ContractParseError> &errors) {

    ContractKind kind;

    if (line.starts_with("requires:")) {
        kind = ContractKind::Requires;
        line = line.substr(9).ltrim(); // Skip "requires:" and whitespace
    } else if (line.starts_with("ensures:")) {
        kind = ContractKind::Ensures;
        line = line.substr(8).ltrim(); // Skip "ensures:" and whitespace
    } else {
        // Not a recognized annotation for Slice 1 -- skip silently.
        // Future slices will handle loop_invariant, assigns, etc.
        return std::nullopt;
    }

    // Parse the expression.
    auto exprResult = parseContractExpr(line.str(), sourceLine);
    if (!exprResult.success()) {
        errors.insert(errors.end(), exprResult.errors.begin(),
                      exprResult.errors.end());
        return std::nullopt;
    }

    return ContractAnnotation{kind, std::move(exprResult.expr), sourceLine};
}

const clang::FunctionDecl *
ContractParser::findAssociatedFunction(
    const clang::RawComment *comment) const {
    auto &sm = context_.getSourceManager();
    // Get the line after the comment ends.
    unsigned commentEndLine =
        sm.getPresumedLineNumber(comment->getEndLoc());

    // Search for the first FunctionDecl starting after the comment.
    auto *tu = context_.getTranslationUnitDecl();
    for (auto *decl : tu->decls()) {
        auto *fn = llvm::dyn_cast<clang::FunctionDecl>(decl);
        if (!fn)
            continue;

        unsigned fnLine = sm.getPresumedLineNumber(fn->getLocation());
        // The function should be on the same line or the line after the
        // comment.
        if (fnLine >= commentEndLine && fnLine <= commentEndLine + 1)
            return fn;
    }
    return nullptr;
}

} // namespace arcanum
