import pgPromise, { IDatabase, ITask } from "pg-promise";
import { MaxMarginalRelevanceSearchOptions, VectorStore } from "./base.js";
import { Embeddings } from "../embeddings/base.js";
import { Document } from "../document.js";
import { maximalMarginalRelevance } from "../util/math.js";
import { isFloat, isInt, isString } from "../util/types.js";

export type FilterValue = string | number;

/**
 * pgp is a factory. It's pure (I think).
 */
const pgp = /* #__PURE__ */ pgPromise();

export type TextSearchValue = {
  query: string;
  type?: "plain" | "phrase" | "websearch";
  config?: string;
};

export type OnCondition = {
  left: string;
  right: string;
  operator?: "=" | "<>" | ">" | ">=" | "<" | "<=";
};

export type JoinStatement = {
  op:
    | "JOIN"
    | "LEFT JOIN"
    | "RIGHT JOIN"
    | "FULL JOIN"
    | "CROSS JOIN"
    | "INNER JOIN";
  table: string;
  on: OnCondition[]; // array to support multiple conditions
};

export type ComparisonOperator =
  | { $eq: FilterValue }
  | { $gt: FilterValue }
  | { $gte: FilterValue }
  | { $lt: FilterValue }
  | { $lte: FilterValue }
  | { $not: FilterValue }
  | { $textSearch: TextSearchValue };

export type LogicalOperator = { $and: PGFilter[] } | { $or: PGFilter[] };

export type KeyValueFilter = {
  [key: string]: FilterValue | ComparisonOperator;
};

export type PGFilter = KeyValueFilter | LogicalOperator;

export type PGFilterWithJoin =
  | {
      metadataFilter?: never;
      columnFilter?: PGFilter;
      join?: JoinStatement | JoinStatement[];
    }
  | {
      metadataFilter?: PGFilter;
      columnFilter?: never;
      join?: JoinStatement | JoinStatement[];
    };

export type Metric = "cosine" | "l2" | "manhattan" | "inner_product";

interface CoreColumns {
  id: string;
  pageContentColumn: string;
  metadata: Record<string, unknown>;
  embedding: number[]; // Replace with the actual data type you use
}

// Use mapped types for extra columns
type ExtraColumns<T extends Record<string, unknown>> = {
  [K in keyof T]: T[K];
};

// Combine core and extra columns
type TableShape<T extends Record<string, unknown> = Record<string, unknown>> =
  CoreColumns & ExtraColumns<T>;

export type ExtensionOpts<T extends Record<string, unknown>> = {
  /**
   * The metric to use for similarity search.
   */
  metric?: Metric;

  /**
   * The number of dimensions of the embeddings.
   */
  dims?: number;

  /**
   * The PG instance to use.
   */
  pgDb: IDatabase<TableShape<T>>;
};

const ComparisonMap = {
  $eq: "=",
  $lt: "<",
  $lte: "<=",
  $gt: ">",
  $gte: ">=",
  $not: "<>",
  $textSearch: "@@",
} as const;
type ComparisonMapKey = keyof typeof ComparisonMap;

const LogicalMap = {
  $and: "AND",
  $or: "OR",
} as const;
type LogicalMapKey = keyof typeof LogicalMap;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type ColumnValue = { [K: string]: any };

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type PGQuery<R> = (t: IDatabase<any> | ITask<any>) => Promise<R>;

export type HNSWIndexStatementOpts = {
  /**
   * the max number of connections per layer (16 by default)
   */
  m?: number;

  /**
   * the size of the dynamic candidate list for constructing the graph (64 by default)
   */
  efConstruction?: number;

  /**
   * Influences the trade-off between query accuracy (recall) and speed. Only pgembedding supports this option.
   */
  efSearch?: number;
};

/**
 * Abstract class for Postgres embedding extensions. This class is used to
 * specify the Postgres extension to use for similarity search. The default
 * extension is pgvector, but pgembedding can also be used.
 */
export abstract class PostgresEmbeddingExtension<
  Columns extends Record<string, unknown> = Record<string, unknown>
> {
  selectedMetric: Metric;

  /**
   * The number of dimensions of the embeddings.
   */
  dims: number;

  pgInstance: IDatabase<TableShape<Columns>>;

  constructor(extensionOpts: ExtensionOpts<Columns>) {
    const metric = extensionOpts.metric ?? "cosine";
    this.validateSelectedMetric(metric);
    this.selectedMetric = metric;
    this.dims = extensionOpts.dims ?? 1536; // defaults to OpenAI 1536 embedding dims
    this.pgInstance = extensionOpts.pgDb;
  }

  abstract allowedMetrics(): Metric[];

  /**
   * Validate the selected metric. Check if it is one of the allowed metrics.
   * pgvector supports cosine, l2, and inner_product. pgembedding supports
   * cosine, l2, and manhattan.
   * @param metric - The metric to validate.
   */
  private validateSelectedMetric(metric: Metric) {
    if (!this.allowedMetrics().includes(metric)) {
      throw new Error(
        `Invalid metric: ${metric}. Allowed metrics are: ${this.allowedMetrics().join(
          ", "
        )}`
      );
    }
  }

  /**
   * Build the SQL statement to fetch rows from the database.
   * @param returns - The columns to return from the database.
   * @param tableName - The name of the table to fetch rows from.
   * @returns {string} The SQL statement to fetch rows from the database.
   */
  abstract buildFetchRowsStatement(
    vector: number[],
    tableName: string,
    returns: string[],
    disambiguate?: boolean
  ): string;

  /**
   * Build the SQL statement to ensure the extension is installed in the
   * database.
   */
  abstract buildEnsureExtensionStatement(): string;

  /**
   * Build the SQL statement to get the data type of the embedding column
   * the chosen embedding extension uses.
   */
  abstract buildDataType(): string;

  /**
   * Build the SQL statement to insert a vector into the database.
   * @param vector - The vector to insert into the database.
   */
  abstract buildInsertionVector(vector: number[]): string;

  /**
   * Build the SQL statement to create an HNSW index on the embedding column.
   * pgvector (post v0.5.0) and pgembedding both support HNSW indexes, but it's
   * still the simplest form of indexing (i.e. no partitioned table index, partial
   * index, etc etc.)
   * @param tableName - The name of the table to create the index on.
   * @param indexOpts - Options for the index.
   */
  abstract buildHNSWIndexStatement(
    indexName: string,
    tableName: string,
    columnName: string,
    indexOpts: { m?: number; efConstruction?: number; efSearch?: number }
  ): string;

  abstract runQueryWrapper<R>(
    dbInstance: IDatabase<TableShape<Columns>>,
    query: PGQuery<R>
  ): Promise<R>;
}

export class PGEmbeddingExt<
  Columns extends Record<string, unknown> = Record<string, unknown>
> extends PostgresEmbeddingExtension<Columns> {
  allowedMetrics(): Metric[] {
    return ["cosine", "l2", "manhattan"];
  }

  buildFetchRowsStatement(
    vector: number[],
    tableName: string,
    returns: string[],
    disambiguate?: boolean
  ): string {
    let arrow;
    switch (this.selectedMetric) {
      case "cosine":
        arrow = "<=>";
        break;
      case "l2":
        arrow = "<->";
        break;
      case "manhattan":
        arrow = "<~>";
        break;
      default:
        throw new Error("Invalid metric");
    }
    let selectStatement;
    if (disambiguate) {
      selectStatement =
        returns.length > 0
          ? returns.map((col) => `${tableName}.${col} AS ${col}`).join(", ")
          : "*";
    } else {
      selectStatement = returns.length > 0 ? returns.join(", ") : "*";
    }

    return pgp.as.format(
      `SELECT ${selectStatement}, embedding ${arrow} $1:raw AS "_distance" FROM ${tableName}`,
      [`array[${vector.join(",")}]`]
    );
  }

  buildEnsureExtensionStatement(): string {
    return "CREATE EXTENSION IF NOT EXISTS embedding;";
  }

  buildDataType(): string {
    return "REAL[]";
  }

  buildInsertionVector(vector: number[]): string {
    return `{${vector.join(",")}}`;
  }

  buildHNSWIndexStatement(
    indexName: string,
    tableName: string,
    columnName: string,
    {
      m = 16,
      efConstruction = 64,
      efSearch,
    }: { m?: number; efConstruction?: number; efSearch?: number }
  ): string {
    const opts = [`dims = ${this.dims}`];
    if (m) opts.push(`m = ${m}`);
    if (efConstruction) opts.push(`efconstruction = ${efConstruction}`);
    if (efSearch) opts.push(`efsearch = ${efSearch}`);

    let ops;
    switch (this.selectedMetric) {
      case "cosine":
        ops = " ann_cos_ops";
        break;
      case "l2":
        ops = "";
        break;
      case "manhattan":
        ops = " ann_manhattan_ops";
        break;
      default:
        throw new Error("Invalid metric");
    }
    return `CREATE INDEX IF NOT EXISTS ${indexName} ON ${tableName} USING hnsw(${columnName}${ops}) WITH (${opts.join(
      ", "
    )});`;
  }

  /**
   * For some reason, pgembedding requires you to set enable_seqscan to off
   * when using it with hnsw indexes. This function wraps the query with a
   * transaction that sets enable_seqscan to off.
   * @param PGInstance
   * @param query
   * @returns
   */
  runQueryWrapper<R>(
    dbInstance: IDatabase<TableShape<Columns>>,
    query: PGQuery<R>
  ): Promise<R> {
    return dbInstance.tx(async (t) => {
      await t.none("SET LOCAL enable_seqscan = off;");
      return query(t);
    });
  }
}

export class PGVectorExt<
  Columns extends Record<string, unknown> = Record<string, unknown>
> extends PostgresEmbeddingExtension<Columns> {
  allowedMetrics(): Metric[] {
    return ["cosine", "l2", "inner_product"];
  }

  buildFetchRowsStatement(
    vector: number[],
    tableName: string,
    returns: string[],
    disambiguate?: boolean
  ): string {
    let embeddingStatement;
    switch (this.selectedMetric) {
      case "cosine":
        embeddingStatement = "1 - (embedding <=> $1::vector)";
        break;
      case "l2":
        embeddingStatement = "embedding <-> $1::vector";
        break;
      case "inner_product":
        embeddingStatement = "(embedding <#> $1::vector) * -1";
        break;
      default:
        throw new Error("Invalid metric");
    }
    let selectStatement;
    if (disambiguate) {
      selectStatement =
        returns.length > 0
          ? returns.map((col) => `${tableName}.${col} AS ${col}`).join(", ")
          : "*";
    } else {
      selectStatement = returns.length > 0 ? returns.join(", ") : "*";
    }
    const queryStr = `[${vector.join(",")}]`;
    return pgp.as.format(
      `SELECT ${selectStatement}, ${embeddingStatement} AS "_distance" FROM ${tableName}`,
      [queryStr]
    );
  }

  buildEnsureExtensionStatement(): string {
    return "CREATE EXTENSION IF NOT EXISTS vector;";
  }

  buildDataType(): string {
    return `vector(${this.dims})`;
  }

  buildInsertionVector(vector: number[]): string {
    return `[${vector.join(",")}]`;
  }

  buildHNSWIndexStatement(
    indexName: string,
    tableName: string,
    columnName: string,
    {
      m = 16,
      efConstruction = 64,
      efSearch,
    }: { m?: number; efConstruction?: number; efSearch?: number }
  ): string {
    let efSearchStr = "";
    const opts = [];
    if (m) opts.push(`m = ${m}`);
    if (efConstruction) opts.push(`ef_construction = ${efConstruction}`);
    if (efSearch) {
      efSearchStr = `\nSET hnsw.ef_search = ${efSearch};`;
    }

    let ops;
    switch (this.selectedMetric) {
      case "cosine":
        ops = " vector_cosine_ops";
        break;
      case "l2":
        ops = " vector_l2_ops";
        break;
      case "inner_product":
        ops = " vector_ip_ops";
        break;
      default:
        throw new Error("Invalid metric");
    }
    return `CREATE INDEX ${indexName} ON ${tableName} USING hnsw(${columnName}${ops})${
      opts.length > 0 ? ` WITH (${opts.join(", ")})` : ""
    };${efSearchStr}`;
  }

  runQueryWrapper<R>(
    dbInstance: IDatabase<TableShape<Columns>>,
    query: PGQuery<R>
  ): Promise<R> {
    return query(dbInstance);
  }
}

/**
 * A search result returned by the PGVectorStore.
 */
export type SearchResult = {
  pageContent: string;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  metadata: Record<string, any>;
  embedding: number[];
  _distance: number;
};

/**
 * A column in the database. Used for extra columns.
 */
export type Column = {
  type: string;
  name: string;
  returned: boolean;
  notNull?: boolean;
  references?:
    | string
    | {
        table: string;
        column: string;
      };
};

export interface PGVectorStoreArgs<
  T extends Record<string, unknown> = Record<string, unknown>
> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  pgDb: IDatabase<TableShape<T>> | Record<string, any> | string;
  useHnswIndex: boolean;
  tableName?: string;
  pageContentColumn?: string;
  pgExtensionOpts?:
    | { type: "pgvector" | "pgembedding"; dims?: number; metric?: Metric }
    | PostgresEmbeddingExtension;
  extraColumns?: Column[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
function isIDatabase(obj: any): obj is IDatabase<any> {
  // Here, define what properties you expect IDatabase to have.
  // For example, assume that IDatabase should have a 'one' method:
  return obj && typeof obj.one === "function";
}

export class PGVectorStore<
  T extends Record<string, unknown> = Record<string, unknown>
> extends VectorStore {
  declare FilterType: PGFilterWithJoin;

  pgInstance: IDatabase<TableShape<T>>;

  tableName: string;

  pageContentColumn: string;

  pgExtension: PostgresEmbeddingExtension;

  extraColumns: Column[];

  useHnswIndex: boolean;

  constructor(embeddings: Embeddings, args: PGVectorStoreArgs) {
    super(embeddings, args);
    this.embeddings = embeddings;
    this.useHnswIndex = args.useHnswIndex;
    this.tableName = args.tableName ?? "documents";
    this.pageContentColumn = args.pageContentColumn ?? "content";
    this.extraColumns = args.extraColumns ?? [];

    if (
      typeof args.pgDb === "string" ||
      (typeof args.pgDb === "object" &&
        ("host" in args.pgDb || "database" in args.pgDb))
    ) {
      this.pgInstance = pgp(args.pgDb) as IDatabase<TableShape<T>>;
    } else if (isIDatabase(args.pgDb)) {
      this.pgInstance = args.pgDb as IDatabase<TableShape<T>>;
    } else {
      throw new Error("Invalid pg-promise argument");
    }

    if (
      typeof args.pgExtensionOpts === "object" &&
      "type" in args.pgExtensionOpts
    ) {
      if (args.pgExtensionOpts.type === "pgvector") {
        this.pgExtension = new PGVectorExt({
          pgDb: this.pgInstance,
          dims: args.pgExtensionOpts.dims,
          metric: args.pgExtensionOpts.metric,
        });
      } else if (args.pgExtensionOpts.type === "pgembedding") {
        this.pgExtension = new PGEmbeddingExt({
          pgDb: this.pgInstance,
          dims: args.pgExtensionOpts.dims,
          metric: args.pgExtensionOpts.metric,
        });
      }
    } else if (
      args.pgExtensionOpts != null &&
      // eslint-disable-next-line no-instanceof/no-instanceof
      args.pgExtensionOpts instanceof PostgresEmbeddingExtension
    ) {
      this.pgExtension = args.pgExtensionOpts;
    } else {
      this.pgExtension = new PGVectorExt({ pgDb: this.pgInstance });
    }
  }

  _vectorstoreType(): string {
    return "PG";
  }

  /**
   * Functions that executes the SQL queries. Can be used to modify how the queries
   * are being executed by inheriting classes. Very very useful when combined
   * with, let's say, transactions to leverage Postgres's RLS feature
   * @param query
   * @returns { PGT.QueryBuilder | PGT.Raw }
   */
  protected runQuery<R>(query: PGQuery<R>) {
    if (this.useHnswIndex) {
      return this.pgExtension.runQueryWrapper(this.pgInstance, query);
    } else {
      return query(this.pgInstance);
    }
  }

  async addVectors(
    vectors: number[][],
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    documents: Document<Record<string, any>>[],
    options?: {
      extraColumns: (ColumnValue | null)[];
    }
  ): Promise<void> {
    await this.ensureTableInDatabase();
    const rows = vectors.map((embedding, idx) => {
      const extraColumns = Object.entries(
        options?.extraColumns[idx] ?? {}
      ).reduce((acc, [key, value]) => {
        // check if key is in this.extraColumns
        const column = this.extraColumns.find(({ name }) => name === key);
        if (column == null) return acc;

        acc[key] = value;
        return acc;
      }, {} as ColumnValue);

      const embeddingString = this.pgExtension.buildInsertionVector(embedding);
      const documentRow = {
        [this.pageContentColumn]: documents[idx].pageContent,
        embedding: embeddingString,
        metadata: documents[idx].metadata,
        ...extraColumns,
      };

      return documentRow;
    });
    await this.runQuery((database) => {
      const columnSet = new pgp.helpers.ColumnSet(Object.keys(rows[0]), {
        table: this.tableName,
      });
      const insertQuery = pgp.helpers.insert(rows, columnSet);
      return database.none(insertQuery);
    });
  }

  async addDocuments(
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    documents: Document<Record<string, any>>[],
    options?: {
      extraColumns: (ColumnValue | null)[];
    }
  ): Promise<void | string[]> {
    const texts = documents.map(({ pageContent }) => pageContent);
    return this.addVectors(
      await this.embeddings.embedDocuments(texts),
      documents,
      options
    );
  }

  async ensureTableInDatabase(): Promise<void> {
    await this.pgInstance.none(
      this.pgExtension.buildEnsureExtensionStatement()
    );
    await this.pgInstance.none('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";');

    const columns = [
      "id uuid NOT NULL DEFAULT uuid_generate_v4() PRIMARY KEY",
      `${this.pageContentColumn} text`,
      "metadata jsonb",
      `embedding ${this.pgExtension.buildDataType()}`,
    ];

    const extraColumns = this.extraColumns.map(
      ({ name, type, notNull, references }) => {
        let refString = "";
        if (references) {
          if (typeof references === "string") {
            refString = ` REFERENCES ${references}`;
          } else {
            refString = ` REFERENCES ${references.table} (${references.column})`;
          }
        }
        return `${name} ${type}${notNull ? " NOT NULL" : ""}${refString}`;
      }
    );

    await this.pgInstance.none(`
      CREATE TABLE IF NOT EXISTS ${this.tableName} (${[
      ...columns,
      ...extraColumns,
    ].join(", ")});
    `);
  }

  private buildJoinStatement(statement: JoinStatement): string {
    const { op, table, on } = statement;
    if (
      ![
        "JOIN",
        "LEFT JOIN",
        "RIGHT JOIN",
        "FULL JOIN",
        "CROSS JOIN",
        "INNER JOIN",
      ].includes(op)
    ) {
      throw new Error(`Invalid join statement: ${op}`);
    }

    const onConditions = on
      .map(
        (condition) =>
          `${condition.left} ${condition.operator || "="} ${condition.right}`
      )
      .join(" AND ");

    return `${op} ${table} ON ${onConditions}`;
  }

  private async fetchRows(
    query: number[],
    k: number,
    filter?: this["FilterType"] | undefined,
    returnedColumns?: string[]
  ): Promise<SearchResult[]> {
    const { metadataFilter, columnFilter, join: joinStatement } = filter ?? {};
    let shouldDisambiguate = false;
    if (joinStatement) {
      shouldDisambiguate = true;
    }
    let filterType: "metadata" | "column" | undefined;

    let joinStatements: string[];
    if (Array.isArray(joinStatement)) {
      joinStatements = joinStatement.map((statement) =>
        this.buildJoinStatement(statement)
      );
    } else {
      joinStatements = joinStatement
        ? [this.buildJoinStatement(joinStatement)]
        : [];
    }

    if (metadataFilter && columnFilter) {
      throw new Error("Cannot have both metadataFilter and columnFilter");
    } else if (metadataFilter) {
      filterType = "metadata";
    } else if (columnFilter) {
      filterType = "column";
    }

    const selectedColumns = [
      ...(returnedColumns ?? []),
      ...this.extraColumns
        .filter(({ returned }) => returned)
        .map(({ name }) => name),
    ];
    const queryStr = [
      this.pgExtension.buildFetchRowsStatement(
        query,
        this.tableName,
        ["id", this.pageContentColumn, "metadata", ...selectedColumns],
        shouldDisambiguate
      ),
      ...joinStatements,
      this.buildSqlFilterStr(metadataFilter ?? columnFilter, filterType),
      pgp.as.format(`ORDER BY "_distance" LIMIT $1;`, [k]),
    ]
      .filter((x) => x != null && x !== "")
      .join(" ");

    const results = await this.runQuery((database) => database.any(queryStr));

    if (results && results.length > 0 && "embedding" in results[0]) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      results.forEach((_: any, idx: number) => {
        if (typeof results[idx].embedding === "string") {
          try {
            results[idx].embedding = JSON.parse(results[idx].embedding);
          } catch (error) {
            throw new Error("Error parsing embedding");
          }
        }
      });
    }

    return results as SearchResult[];
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: this["FilterType"] | undefined
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
  ): Promise<[Document<Record<string, any>>, number][]> {
    const rows = await this.fetchRows(query, k, filter);
    return rows.map((row) => [
      new Document({
        pageContent: row[this.pageContentColumn as keyof typeof row] as string,
        metadata: row.metadata,
      }),
      row._distance,
    ]);
  }

  async maxMarginalRelevanceSearch(
    query: string,
    options: MaxMarginalRelevanceSearchOptions<this["FilterType"]>
  ): Promise<Document[]> {
    const { k, fetchK = 20, lambda = 0.7, filter } = options;
    const queryEmbedding = await this.embeddings.embedQuery(query);
    const results = await this.fetchRows(queryEmbedding, fetchK, filter, [
      "embedding",
    ]);

    const embeddings = results.map((result) => result.embedding);
    const mmrIndexes = maximalMarginalRelevance(
      queryEmbedding,
      embeddings,
      lambda,
      k
    );
    return mmrIndexes
      .filter((idx) => idx !== -1)
      .map((idx) => {
        const result = results[idx];
        return new Document({
          pageContent: result.pageContent,
          metadata: result.metadata,
        });
      });
  }

  /**
   * Build the HNSW index on the embedding column. Should be
   * called before doing similarity search if you plan to have
   * a large number of rows in the table.
   * @param buildIndexOpt
   */
  async buildIndex(
    indexName: string,
    {
      m,
      efConstruction,
      efSearch,
    }: {
      m?: number;
      efConstruction?: number;
      efSearch?: number;
    } = {}
  ): Promise<void> {
    await this.ensureTableInDatabase();
    await this.pgInstance.none(
      this.pgExtension.buildHNSWIndexStatement(
        indexName,
        this.tableName,
        "embedding",
        {
          m,
          efConstruction,
          efSearch,
        }
      )
    );
  }

  async dropIndex(indexName: string): Promise<void> {
    await this.ensureTableInDatabase();
    await this.pgInstance.none(`DROP INDEX IF EXISTS ${indexName};`);
  }

  buildTextSearchStatement(param: TextSearchValue, column: string) {
    const { query, type, config } = param;
    const lang = `${config ?? "simple"}`;
    let queryOp = "to_tsquery";
    if (type) {
      switch (type) {
        case "plain":
          queryOp = "plainto_tsquery";
          break;
        case "phrase":
          queryOp = "phraseto_tsquery";
          break;
        case "websearch":
          queryOp = "websearch_to_tsquery";
          break;
        default:
          throw new Error("Invalid text search type");
      }
    }

    return pgp.as.format(`to_tsvector($1, ${column}) @@ ${queryOp}($1, $2)`, [
      lang,
      query,
    ]);
  }

  /**
   * Build the SQL filter string from the filter object.
   * @param filter - The filter object
   * @returns
   */
  buildSqlFilterStr(
    filter?: PGFilter,
    type: "metadata" | "column" = "metadata"
  ) {
    if (filter == null) return null;

    const buildClause = (
      key: string,
      operator: ComparisonMapKey,
      value: string | number | TextSearchValue
    ): string => {
      const op = operator;
      const compRaw = ComparisonMap[op];
      const myValue =
        typeof value === "object" && "query" in value ? value.query : value;

      let typeCast;
      let arrow;

      if (isString(myValue)) {
        typeCast = "::text";
        arrow = "->>";
      } else if (isInt(myValue)) {
        typeCast = "::int";
        arrow = "->";
      } else if (isFloat(myValue)) {
        typeCast = "::float";
        arrow = "->";
      } else {
        throw new Error("Data type not supported");
      }
      let columnKey;
      if (key === this.pageContentColumn) {
        columnKey = this.pageContentColumn;
      } else {
        if (type === "column") {
          columnKey = `${key}`;
          typeCast = "";
        } else {
          columnKey = `metadata${arrow}'${key}'`;
        }
      }

      if (op === "$textSearch") {
        return this.buildTextSearchStatement(
          value as TextSearchValue,
          columnKey
        );
      } else {
        return pgp.as.format(`($1:raw)${typeCast} ${compRaw} $2`, [
          columnKey,
          myValue,
        ]);
      }
    };
    const allowedOps = Object.keys(LogicalMap);

    const recursiveBuild = (filterObj: PGFilter): string =>
      Object.entries(filterObj)
        .map(([key, ops]) => {
          if (allowedOps.includes(key)) {
            const logicalParts = (ops as PGFilter[]).map(recursiveBuild);
            const separator = LogicalMap[key as LogicalMapKey];
            return `(${logicalParts.join(` ${separator} `)})`;
          }

          if (typeof ops === "object" && !Array.isArray(ops)) {
            // eslint-disable-next-line @typescript-eslint/no-explicit-any
            return Object.entries(ops as Record<string, any>)
              .map(([opName, value]) => {
                if (!value) return null;
                return buildClause(key, opName as ComparisonMapKey, value);
              })
              .filter(Boolean)
              .join(" AND ");
          }

          return buildClause(key, "$eq", ops);
        })
        .filter(Boolean)
        .join(" AND ");

    const strFilter = `WHERE ${recursiveBuild(filter)}`;

    if (strFilter === "WHERE ") return null;
    return strFilter;
  }
}
