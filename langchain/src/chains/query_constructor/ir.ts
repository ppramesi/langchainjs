import { Document } from "../../document.js";

export type AND = "and";
export type OR = "or";
export type NOT = "not";

export type Operator = AND | OR | NOT;

export type EQ = "eq";
export type NEQ = "neq";
export type LT = "lt";
export type GT = "gt";
export type LTE = "lte";
export type GTE = "gte";
export type IN = "in";
export type NIN = "nin";

export type Comparator = EQ | NEQ | LT | GT | LTE | GTE | IN | NIN;

export const Operators: { [key: string]: Operator } = {
  and: "and",
  or: "or",
  not: "not",
};

export const Comparators: { [key: string]: Comparator } = {
  eq: "eq",
  neq: "neq",
  lt: "lt",
  gt: "gt",
  lte: "lte",
  gte: "gte",
  in: "in",
  nin: "nin",
};

export type FunctionFilter = (document: Document) => boolean;

export type VisitorResult =
  | VisitorOperationResult
  | VisitorComparisonResult
  | VisitorStructuredQueryResult;

export type VisitorOperationResult =
  | {
      [operator: string]: VisitorResult[];
    }
  | FunctionFilter;

export type VisitorComparisonResult =
  | {
      [attr: string]: {
        [comparator: string]: string | number | string[] | number[];
      };
    }
  | FunctionFilter;

export type VisitorStructuredQueryResult =
  | {
      filter?:
        | VisitorStructuredQueryResult
        | VisitorComparisonResult
        | VisitorOperationResult;
    }
  | {
      filter?: FunctionFilter;
    };

export abstract class Visitor {
  abstract allowedOperators: Operator[];

  abstract allowedComparators: Comparator[];

  abstract visitOperation(operation: Operation): VisitorOperationResult;

  abstract visitComparison(comparison: Comparison): VisitorComparisonResult;

  abstract visitStructuredQuery(
    structuredQuery: StructuredQuery
  ): VisitorStructuredQueryResult;
}

export abstract class Expression {
  abstract exprName: "Operation" | "Comparison" | "StructuredQuery";

  accept(visitor: Visitor) {
    if (this.exprName === "Operation") {
      return visitor.visitOperation(this as unknown as Operation);
    } else if (this.exprName === "Comparison") {
      return visitor.visitComparison(this as unknown as Comparison);
    } else if (this.exprName === "StructuredQuery") {
      return visitor.visitStructuredQuery(this as unknown as StructuredQuery);
    } else {
      throw new Error("Unknown Expression type");
    }
  }
}

export abstract class FilterDirective extends Expression {}

export class Comparison extends FilterDirective {
  exprName = "Comparison" as const;

  constructor(
    public comparator: Comparator,
    public attribute: string,
    public value: string | number | string[] | number[]
  ) {
    super();
  }
}

export class Operation extends FilterDirective {
  exprName = "Operation" as const;

  constructor(public operator: Operator, public args?: FilterDirective[]) {
    super();
  }
}

export class StructuredQuery extends Expression {
  exprName = "StructuredQuery" as const;

  constructor(public query: string, public filter?: FilterDirective) {
    super();
  }
}
