import type { ESTree } from "meriyah";
import { NodeHandler, ASTParser } from "./base.js";
import { MemberExpressionType } from "./types.js";
import { IdentifierHandler } from "./identifier_handler.js";

export class MemberExpressionHandler extends NodeHandler {
  async accepts(node: ESTree.Node): Promise<ESTree.MemberExpression | boolean> {
    if (ASTParser.isMemberExpression(node)) {
      return node;
    } else {
      return false;
    }
  }

  async handle(node: ESTree.MemberExpression): Promise<MemberExpressionType> {
    if (!this.parentHandler) {
      throw new Error(
        "ArrayLiteralExpressionHandler must have a parent handler"
      );
    }
    const { object, property } = node;
    let prop: string;
    if (ASTParser.isIdentifier(property)) {
      const identifierHandler = new IdentifierHandler(this.parentHandler);
      const handledIdentifier = await identifierHandler.handle(property);
      prop = handledIdentifier.value;
    } else if (ASTParser.isStringLiteral(property)) {
      prop = (`${property.value}` as string).replace(
        /^["'](.+(?=["']$))["']$/,
        "$1"
      );
    } else {
      throw new Error("Invalid property key type");
    }
    let identifier: string;
    if (ASTParser.isIdentifier(object)) {
      const identifierHandler = new IdentifierHandler(this.parentHandler);
      const handledIdentifier = await identifierHandler.handle(object);
      identifier = handledIdentifier.value;
    } else if (ASTParser.isLiteral(object)) {
      identifier = (`${object.value}` as string).replace(
        /^["'](.+(?=["']$))["']$/,
        "$1"
      );
    } else {
      throw new Error("Invalid object type");
    }
    if (object.type !== "Identifier" && object.type !== "Literal") {
      throw new Error("ArrayExpression is not supported");
    }

    return { type: "member_expression", identifier, property: prop };
  }
}
