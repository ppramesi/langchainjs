/* eslint-disable no-process-env */
import { Document } from "../../document.js";
import {
  test,
  expect,
  beforeAll,
  beforeEach,
  afterAll,
  afterEach,
} from "@jest/globals";
import pgPromise, { IDatabase } from "pg-promise";
import { v4 } from "uuid";
import { FakeEmbeddings, SyntheticEmbeddings } from "../../embeddings/fake.js";
import { PGVectorStore } from "../pg.js";

class NormalizedSyntheticEmbeddings extends SyntheticEmbeddings {
  normalizeVector(vector: number[]): number[] {
    let norm = Math.sqrt(vector.reduce((acc, val) => acc + val * val, 0));
    if (norm === 0) {
      return vector;
    }

    return vector.map((val) => val / norm);
  }

  /**
   * Generates a synthetic embedding for a document. The document is
   * converted into chunks, a numerical value is calculated for each chunk,
   * and an array of these values is returned as the embedding.
   * @param document The document to generate an embedding for.
   * @returns A promise that resolves with a synthetic embedding for the document.
   */
  async embedQuery(document: string): Promise<number[]> {
    let doc = document;

    // Only use the letters (and space) from the document, and make them lower case
    doc = doc.toLowerCase().replaceAll(/[^a-z ]/g, "");

    // Pad the document to make sure it has a divisible number of chunks
    const padMod = doc.length % this.vectorSize;
    const padGapSize = padMod === 0 ? 0 : this.vectorSize - padMod;
    const padSize = doc.length + padGapSize;
    doc = doc.padEnd(padSize, " ");

    // Break it into chunks
    const chunkSize = doc.length / this.vectorSize;
    const docChunk = [];
    for (let co = 0; co < doc.length; co += chunkSize) {
      docChunk.push(doc.slice(co, co + chunkSize));
    }

    // Turn each chunk into a number
    const ret: number[] = docChunk.map((s) => {
      let sum = 0;
      // Get a total value by adding the value of each character in the string
      for (let co = 0; co < s.length; co += 1) {
        sum += s === " " ? 0 : s.charCodeAt(co);
      }
      // the only change, since we need the vectors to be normalized for testing cosine distance
      const ret = ((sum % 26) - 13) / 13;
      return ret;
    });

    return this.normalizeVector(ret);
  }
}

/**
 * We're using two different postgres instances for each extension. Should setup with docker,
 * see /examples/src/indexes/vector_stores/pg_vectorstore/pgvector/Dockerfile
 * and /examples/src/indexes/vector_stores/pg_vectorstore/pgembedding/Dockerfile
 * for dockerfile configs.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pgvsPgvector: IDatabase<any>;
// eslint-disable-next-line @typescript-eslint/no-explicit-any
let pgvsPgembedding: IDatabase<any>;

if (
  !process.env.POSTGRES_HOST ||
  !process.env.POSTGRES_PGVECTOR_DB ||
  !process.env.POSTGRES_PGVECTOR_USER ||
  !process.env.POSTGRES_PGVECTOR_PASSWORD ||
  !process.env.POSTGRES_PGVECTOR_PORT
) {
  throw new Error("PGVECTOR environment variables not set");
}

if (
  !process.env.POSTGRES_HOST ||
  !process.env.POSTGRES_PGEMBEDDING_DB ||
  !process.env.POSTGRES_PGEMBEDDING_USER ||
  !process.env.POSTGRES_PGEMBEDDING_PASSWORD ||
  !process.env.POSTGRES_PGEMBEDDING_PORT
) {
  throw new Error("PGEMBEDDING environment variables not set");
}

beforeAll(async () => {
  const pgp = pgPromise();
  pgvsPgvector = pgp({
    host: process.env.POSTGRES_HOST,
    database: process.env.POSTGRES_PGVECTOR_DB,
    user: process.env.POSTGRES_PGVECTOR_USER,
    password: process.env.POSTGRES_PGVECTOR_PASSWORD,
    port: Number(process.env.POSTGRES_PGVECTOR_PORT),
    max: 20,
  });
  pgvsPgembedding = pgp({
    host: process.env.POSTGRES_HOST,
    database: process.env.POSTGRES_PGEMBEDDING_DB,
    user: process.env.POSTGRES_PGEMBEDDING_USER,
    password: process.env.POSTGRES_PGEMBEDDING_PASSWORD,
    port: Number(process.env.POSTGRES_PGEMBEDDING_PORT),
    max: 20,
  });
});

beforeEach(async () => {
  /**
   * 🚨🚨🚨 WARNING WARNING WARNING 🚨🚨🚨
   * We're dropping pg_embeddings and other tables first to make
   * sure the tests are idempotent. This means that if you have
   * a table called pg_embeddings (and similar names) in your
   * database, it will be dropped.
   */
  await Promise.all([
    pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings"),
    pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings_metric_test"),
    pgvsPgvector
      .none("DROP TABLE IF EXISTS pg_embeddings_test_join;")
      .then(() => pgvsPgvector.none("DROP TABLE IF EXISTS some_extra_stuff;")),
    pgvsPgvector
      .none("DROP TABLE IF EXISTS injection_test;")
      .then(() =>
        pgvsPgvector.none("DROP TABLE IF EXISTS injection_some_extra_stuff;")
      ),
    pgvsPgembedding.none("DROP TABLE IF EXISTS pg_embeddings"),
    pgvsPgembedding.none("DROP TABLE IF EXISTS pg_embeddings_metric_test"),
    pgvsPgembedding
      .none("DROP TABLE IF EXISTS pg_embeddings_test_join;")
      .then(() =>
        pgvsPgembedding.none("DROP TABLE IF EXISTS some_extra_stuff;")
      ),
    pgvsPgembedding
      .none("DROP TABLE IF EXISTS injection_test;")
      .then(() =>
        pgvsPgembedding.none("DROP TABLE IF EXISTS injection_some_extra_stuff;")
      ),
  ]);

  const embedding = new FakeEmbeddings();

  const pgvKnexVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgvector,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgvector", dims: 4 },
  });
  const pgeKnexVS = new PGVectorStore(embedding, {
    postgresConnectionOptions: pgvsPgembedding,
    useHnswIndex: true,
    tableName: "pg_embeddings",
    columns: {
      contentColumnName: "content",
    },
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtensionOpts: { type: "pgembedding", dims: 4 },
  });

  await Promise.all([
    pgvKnexVS.ensureTableInDatabase(),
    pgeKnexVS.ensureTableInDatabase(),
  ]);
});

afterEach(async () => {
  /**
   * 🚨🚨🚨 WARNING WARNING WARNING 🚨🚨🚨
   * We're dropping pg_embeddings and other tables first to make
   * sure the tests are idempotent. This means that if you have
   * a table called pg_embeddings (and similar names) in your
   * database, it will be dropped.
   */
  await Promise.all([
    pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings"),
    pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings_metric_test"),
    pgvsPgvector
      .none("DROP TABLE IF EXISTS pg_embeddings_test_join;")
      .then(() => pgvsPgvector.none("DROP TABLE IF EXISTS some_extra_stuff;")),
    pgvsPgvector
      .none("DROP TABLE IF EXISTS injection_test;")
      .then(() =>
        pgvsPgvector.none("DROP TABLE IF EXISTS injection_some_extra_stuff;")
      ),
    pgvsPgembedding.none("DROP TABLE IF EXISTS pg_embeddings"),
    pgvsPgembedding.none("DROP TABLE IF EXISTS pg_embeddings_metric_test"),
    pgvsPgembedding
      .none("DROP TABLE IF EXISTS pg_embeddings_test_join;")
      .then(() =>
        pgvsPgembedding.none("DROP TABLE IF EXISTS some_extra_stuff;")
      ),
    pgvsPgembedding
      .none("DROP TABLE IF EXISTS injection_test;")
      .then(() =>
        pgvsPgembedding.none("DROP TABLE IF EXISTS injection_some_extra_stuff;")
      ),
  ]);
});

afterAll(async () => {
  await Promise.all([pgvsPgvector.$pool.end(), pgvsPgembedding.$pool.end()]);
});

describe("pgvector tests", () => {
  test("Build index pgvector", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    try {
      await pgVS.buildIndex("test_hnsw_index");
      expect(true).toBe(true);
    } catch (error) {
      expect(error).toBe(null);
    } finally {
      await pgVS.dropIndex("test_hnsw_index");
    }
  });

  test("Upsert test", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: false,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    const docs = [
      {
        pageContent: "hello",
        metadata: { b: 1, c: 10 },
      },
      {
        pageContent: "hoooo",
        metadata: { b: 2, c: 9 },
      },
    ];
    const id1 = v4();
    const id2 = v4();

    await pgVS.addDocuments(docs, {
      extraColumns: [{ extra_stuff: "hello 1" }, { extra_stuff: "hello 2" }],
      ids: [id1, id2],
    });

    await pgVS.addDocuments(docs, {
      extraColumns: [{ extra_stuff: "hello 3" }, { extra_stuff: "hello 4" }],
      ids: [id1, id2],
    });

    const results = await pgVS.similaritySearch("hello 1", 2);
    expect(results.map((l) => l.metadata?.extra_stuff)).toEqual([
      "hello 3",
      "hello 4",
    ]);
  });

  test("MMR and Similarity Search Test pgvector", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: "hello 1" },
        { extra_stuff: "hello 2" },
        { extra_stuff: "hello 3" },
        { extra_stuff: "hello 4" },
        { extra_stuff: "hello 5" },
        { extra_stuff: "hello 6" },
        { extra_stuff: "hello 7" },
      ],
    });

    const ssResults = await pgVS.similaritySearch("hello", 7);

    expect(ssResults.length).toBe(7);

    const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
      k: 3,
      fetchK: 7,
    });

    expect(mmrResults.length).toBe(3);
  });

  test("Building WHERE query test pgvector", () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    const queryMetadata = pgVS.buildSqlFilterStr(
      {
        $or: [
          { stuff: { $eq: "hello" } },
          { hello: "stuff" },
          {
            $and: [
              { hello: "stuff" },
              {
                content: {
                  $textSearch: {
                    query: "hello",
                    config: "english",
                    type: "plain",
                  },
                },
              },
            ],
          },
        ],
      },
      "metadata"
    );

    expect(JSON.stringify(queryMetadata)).toBe(
      '["WHERE",{"query":"(((metadata->>$1)::text = $2 OR (metadata->>$3)::text = $4 OR ((metadata->>$5)::text = $6 AND to_tsvector($7, (metadata->>$8)::text) @@ plainto_tsquery($9, $10))))","values":["stuff","hello","hello","stuff","hello","stuff","english","content","english","hello"]}]'
    );

    const queryColumn = pgVS.buildSqlFilterStr(
      {
        $or: [
          { stuff: { $eq: "hello" } },
          { hello: "stuff" },
          {
            $and: [
              { hello: "stuff" },
              {
                content: {
                  $textSearch: {
                    query: "hello",
                    config: "english",
                    type: "plain",
                  },
                },
              },
            ],
          },
        ],
      },
      "column"
    );

    expect(JSON.stringify(queryColumn)).toBe(
      '["WHERE",{"query":"(($1:alias = $2 OR $3:alias = $4 OR ($5:alias = $6 AND to_tsvector($7, $8:alias) @@ plainto_tsquery($9, $10))))","values":["stuff","hello","hello","stuff","hello","stuff","english","content","english","hello"]}]'
    );
  });

  test("MMR and Similarity Search with filter Test pgvector", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: "hello 1" },
        { extra_stuff: "hello 2" },
        { extra_stuff: "hello 3" },
        { extra_stuff: "hello 4" },
        { extra_stuff: "hello 5" },
        { extra_stuff: "hello 6" },
        { extra_stuff: "hello 7" },
      ],
    });

    const ssResults = await pgVS.similaritySearch("hello", 2, {
      metadataFilter: {
        b: { $eq: 1 },
      },
    });

    expect(ssResults.length).toBe(2);

    const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
      k: 2,
      fetchK: 7,
      filter: {
        metadataFilter: {
          b: { $eq: 1 },
        },
      },
    });

    expect(mmrResults.length).toBe(2);
  });

  test("MMR and Similarity Search with filter + join Test pgvector", async () => {
    const embedding = new FakeEmbeddings();
    // create some_other_stuff table with pg-promise
    await pgvsPgvector.none(
      "CREATE TABLE IF NOT EXISTS some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
    );
    // add some data to some_other_stuff table.
    await pgvsPgvector.none(
      "INSERT INTO some_extra_stuff (type) VALUES ('hello'), ('hi'), ('bye')"
    );

    try {
      const pgVS = new PGVectorStore(embedding, {
        postgresConnectionOptions: pgvsPgvector,
        useHnswIndex: true,
        tableName: "pg_embeddings_test_join",
        columns: {
          contentColumnName: "content",
        },
        extraColumns: [
          {
            name: "extra_stuff",
            type: "integer",
            returned: true,
            references: { table: "some_extra_stuff", column: "id" },
          },
        ],
        pgExtensionOpts: { type: "pgvector", dims: 4 },
      });

      await pgVS.ensureTableInDatabase();

      const createdAt = new Date().getTime();

      const docs = [
        {
          pageContent:
            "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
          metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
        },
        {
          pageContent:
            "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
          metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "hello",
          metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "hello",
          metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
        },
        {
          pageContent: "hi",
          metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "bye",
          metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "what's this",
          metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
        },
      ];

      await pgVS.addDocuments(docs, {
        extraColumns: [
          { extra_stuff: 1 },
          { extra_stuff: 1 },
          { extra_stuff: 1 },
          { extra_stuff: 2 },
          { extra_stuff: 2 },
          { extra_stuff: 3 },
          { extra_stuff: 3 },
        ],
      });

      const ssResults = await pgVS.similaritySearch("hello", 3, {
        join: {
          op: "JOIN",
          table: "some_extra_stuff",
          on: [
            {
              left: "pg_embeddings_test_join.extra_stuff",
              right: "some_extra_stuff.id",
              operator: "=",
            },
          ],
        },
        columnFilter: {
          "some_extra_stuff.type": { $eq: "hello" },
        },
      });

      expect(ssResults.length).toBe(3);

      const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
        k: 3,
        fetchK: 7,
        filter: {
          join: {
            op: "JOIN",
            table: "some_extra_stuff",
            on: [
              {
                left: "pg_embeddings_test_join.extra_stuff",
                right: "some_extra_stuff.id",
                operator: "=",
              },
            ],
          },
          columnFilter: {
            "some_extra_stuff.type": { $eq: "hello" },
          },
        },
      });

      expect(mmrResults.length).toBe(3);
    } finally {
      // drop pg_embeddings_test_join table
      await pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings_test_join;");
      // drop some extra_stuff table
      await pgvsPgvector.none("DROP TABLE IF EXISTS some_extra_stuff;");
    }
  });

  test("Text search test pgvector", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: "hello 1" },
        { extra_stuff: "hello 2" },
        { extra_stuff: "hello 3" },
        { extra_stuff: "hello 4" },
        { extra_stuff: "hello 5" },
        { extra_stuff: "hello 6" },
        { extra_stuff: "hello 7" },
      ],
    });

    const results = await pgVS.similaritySearch("This is a long text", 1, {
      columnFilter: {
        content: {
          $textSearch: {
            query: `'multidimensional' & 'spaces'`,
            config: "english",
          },
        },
      },
    });

    expect(results.length).toBe(1);
  });

  test("Metric test pgvector", async () => {
    const embedding = new NormalizedSyntheticEmbeddings({ vectorSize: 10 });
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: false,
      tableName: "pg_embeddings_metric_test",
      columns: {
        contentColumnName: "content",
      },
      pgExtensionOpts: { type: "pgvector", dims: 10 },
    });

    await pgVS.ensureTableInDatabase();

    const docs = [
      {
        pageContent: "aaaaaaaaaaaaaaaaaaaa",
        metadata: { b: 1, c: 10, stuff: "right" },
      },
      {
        pageContent: "poiuytrewqasdfghjklm",
        metadata: { b: 2, c: 9, stuff: "right" },
      },
      {
        pageContent: "mnbvcxzasdfghjkloiuy",
        metadata: { b: 1, c: 9, stuff: "right" },
      },
      {
        pageContent: "alskdjfhvngbthrycisudj",
        metadata: { b: 1, c: 9, stuff: "wrong" },
      },
      {
        pageContent: "eeeeeeeeeeeeeeeeeeee",
        metadata: { b: 2, c: 8, stuff: "right" },
      },
      {
        pageContent: "ffffffffffffffffffff",
        metadata: { b: 3, c: 7, stuff: "right" },
      },
      {
        pageContent: "gggggggggggggggggggg",
        metadata: { b: 4, c: 6, stuff: "right" },
      },
    ];

    const getPageContent = (docs: Document<Record<string, any>>[]) =>
      docs.map((doc) => doc.pageContent);
    await pgVS.addDocuments(docs);
    const resultsA = await pgVS.similaritySearch("aaaaaaaaaaaaaaaaaaaa", 2);
    expect(getPageContent(resultsA)).toContain("aaaaaaaaaaaaaaaaaaaa");
    const resultsB = await pgVS.similaritySearch("poiuytrewqasdfghjklm", 2);
    expect(getPageContent(resultsB)).toContain("poiuytrewqasdfghjklm");
    const resultsC = await pgVS.similaritySearch("mnbvcxzasdfghjkloiuy", 2);
    expect(getPageContent(resultsC)).toContain("mnbvcxzasdfghjkloiuy");
    const resultsD = await pgVS.similaritySearch("alskdjfhvngbthrycisudj", 2);
    expect(getPageContent(resultsD)).toContain("alskdjfhvngbthrycisudj");
  });
});

describe("pgembedding tests", () => {
  test("Build index pgembedding", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgembedding", dims: 4 },
    });

    try {
      await pgVS.buildIndex("test_hnsw_index");
      expect(true).toBe(true);
    } catch (error) {
      expect(error).toBe(null);
    } finally {
      await pgVS.dropIndex("test_hnsw_index");
    }
  });

  test("MMR and Similarity Search Test pgembedding", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgembedding", dims: 4 },
    });

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: "hello 1" },
        { extra_stuff: "hello 2" },
        { extra_stuff: "hello 3" },
        { extra_stuff: "hello 4" },
        { extra_stuff: "hello 5" },
        { extra_stuff: "hello 6" },
        { extra_stuff: "hello 7" },
      ],
    });

    const ssResults = await pgVS.similaritySearch("hello", 7);

    expect(ssResults.length).toBe(7);

    const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
      k: 3,
      fetchK: 7,
    });

    expect(mmrResults.length).toBe(3);
  });

  test("Building WHERE query test pgembedding", () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgembedding", dims: 4 },
    });

    const queryMetadata = pgVS.buildSqlFilterStr(
      {
        $or: [
          { stuff: { $eq: "hello" } },
          { hello: "stuff" },
          {
            $and: [
              { hello: "stuff" },
              {
                content: {
                  $textSearch: {
                    query: "hello",
                    config: "english",
                    type: "plain",
                  },
                },
              },
            ],
          },
        ],
      },
      "metadata"
    );

    expect(JSON.stringify(queryMetadata)).toBe(
      '["WHERE",{"query":"(((metadata->>$1)::text = $2 OR (metadata->>$3)::text = $4 OR ((metadata->>$5)::text = $6 AND to_tsvector($7, (metadata->>$8)::text) @@ plainto_tsquery($9, $10))))","values":["stuff","hello","hello","stuff","hello","stuff","english","content","english","hello"]}]'
    );

    const queryColumn = pgVS.buildSqlFilterStr(
      {
        $or: [
          { stuff: { $eq: "hello" } },
          { hello: "stuff" },
          {
            $and: [
              { hello: "stuff" },
              {
                content: {
                  $textSearch: {
                    query: "hello",
                    config: "english",
                    type: "plain",
                  },
                },
              },
            ],
          },
        ],
      },
      "column"
    );

    expect(JSON.stringify(queryColumn)).toBe(
      '["WHERE",{"query":"(($1:alias = $2 OR $3:alias = $4 OR ($5:alias = $6 AND to_tsvector($7, $8:alias) @@ plainto_tsquery($9, $10))))","values":["stuff","hello","hello","stuff","hello","stuff","english","content","english","hello"]}]'
    );
  });

  test("MMR and Similarity Search with filter Test pgembedding", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgembedding", dims: 4 },
    });

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: "hello 1" },
        { extra_stuff: "hello 2" },
        { extra_stuff: "hello 3" },
        { extra_stuff: "hello 4" },
        { extra_stuff: "hello 5" },
        { extra_stuff: "hello 6" },
        { extra_stuff: "hello 7" },
      ],
    });

    const ssResults = await pgVS.similaritySearch("hello", 2, {
      metadataFilter: {
        b: { $eq: 1 },
      },
    });

    expect(ssResults.length).toBe(2);

    const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
      k: 2,
      fetchK: 7,
      filter: {
        metadataFilter: {
          b: { $eq: 1 },
        },
      },
    });

    expect(mmrResults.length).toBe(2);
  });

  test("MMR and Similarity Search with filter + join Test pgembedding", async () => {
    const embedding = new FakeEmbeddings();
    // create some_other_stuff table with pg-promise
    await pgvsPgembedding.none(
      "CREATE TABLE IF NOT EXISTS some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
    );
    // add some data to some_other_stuff table.
    await pgvsPgembedding.none(
      "INSERT INTO some_extra_stuff (type) VALUES ('hello'), ('hi'), ('bye')"
    );

    try {
      const pgVS = new PGVectorStore(embedding, {
        postgresConnectionOptions: pgvsPgembedding,
        useHnswIndex: true,
        tableName: "pg_embeddings_test_join",
        columns: {
          contentColumnName: "content",
        },
        extraColumns: [
          {
            name: "extra_stuff",
            type: "integer",
            returned: true,
            references: { table: "some_extra_stuff", column: "id" },
          },
        ],
        pgExtensionOpts: { type: "pgembedding", dims: 4 },
      });

      await pgVS.ensureTableInDatabase();

      const createdAt = new Date().getTime();

      const docs = [
        {
          pageContent:
            "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
          metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
        },
        {
          pageContent:
            "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
          metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "hello",
          metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "hello",
          metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
        },
        {
          pageContent: "hi",
          metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "bye",
          metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
        },
        {
          pageContent: "what's this",
          metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
        },
      ];

      await pgVS.addDocuments(docs, {
        extraColumns: [
          { extra_stuff: 1 },
          { extra_stuff: 1 },
          { extra_stuff: 1 },
          { extra_stuff: 2 },
          { extra_stuff: 2 },
          { extra_stuff: 3 },
          { extra_stuff: 3 },
        ],
      });

      const ssResults = await pgVS.similaritySearch("hello", 3, {
        join: {
          op: "JOIN",
          table: "some_extra_stuff",
          on: [
            {
              left: "pg_embeddings_test_join.extra_stuff",
              right: "some_extra_stuff.id",
              operator: "=",
            },
          ],
        },
        columnFilter: {
          "some_extra_stuff.type": { $eq: "hello" },
        },
      });

      expect(ssResults.length).toBe(3);

      const mmrResults = await pgVS.maxMarginalRelevanceSearch("hello", {
        k: 3,
        fetchK: 7,
        filter: {
          join: {
            op: "JOIN",
            table: "some_extra_stuff",
            on: [
              {
                left: "pg_embeddings_test_join.extra_stuff",
                right: "some_extra_stuff.id",
                operator: "=",
              },
            ],
          },
          columnFilter: {
            "some_extra_stuff.type": { $eq: "hello" },
          },
        },
      });

      expect(mmrResults.length).toBe(3);
    } finally {
      // drop pg_embeddings_test_join table
      await pgvsPgvector.none("DROP TABLE IF EXISTS pg_embeddings_test_join;");
      // drop some extra_stuff table
      await pgvsPgvector.none("DROP TABLE IF EXISTS some_extra_stuff;");
    }
  });

  test("Text search test pgembedding", async () => {
    const embedding = new FakeEmbeddings();
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: true,
      tableName: "pg_embeddings",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
      pgExtensionOpts: { type: "pgembedding", dims: 4 },
    });

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to expand upon the notion of quantum fluff, a theorectical concept where subatomic particles coalesce to form transient multidimensional spaces. Yet, this abstraction holds no real-world application or comprehensible meaning, reflecting a cosmic puzzle.",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent:
          "This is a long text, but it actually means something because vector database does not understand Lorem Ipsum. So I would need to proceed by discussing the echo of virtual tweets in the binary corridors of the digital universe. Each tweet, like a pixelated canary, hums in an unseen frequency, a fascinatingly perplexing phenomenon that, while conjuring vivid imagery, lacks any concrete implication or real-world relevance, portraying a paradox of multidimensional spaces in the age of cyber folklore.",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "wrong", created_at: createdAt },
      },
      {
        pageContent: "hi",
        metadata: { b: 2, c: 8, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "bye",
        metadata: { b: 3, c: 7, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "what's this",
        metadata: { b: 4, c: 6, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: "hello 1" },
        { extra_stuff: "hello 2" },
        { extra_stuff: "hello 3" },
        { extra_stuff: "hello 4" },
        { extra_stuff: "hello 5" },
        { extra_stuff: "hello 6" },
        { extra_stuff: "hello 7" },
      ],
    });

    const results = await pgVS.similaritySearch("This is a long text", 1, {
      columnFilter: {
        content: {
          $textSearch: {
            query: `'multidimensional' & 'spaces'`,
            config: "english",
          },
        },
      },
    });

    expect(results.length).toBe(1);
  });

  test("Metric test pgembedding", async () => {
    const embedding = new NormalizedSyntheticEmbeddings({ vectorSize: 20 });
    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgembedding,
      useHnswIndex: false,
      tableName: "pg_embeddings_metric_test",
      columns: {
        contentColumnName: "content",
      },
      pgExtensionOpts: { type: "pgembedding", dims: 20 },
    });

    await pgVS.ensureTableInDatabase();

    const docs = [
      {
        pageContent: "aaaaaaaaaaaaaaaaaaaa",
        metadata: { b: 1, c: 10, stuff: "right" },
      },
      {
        pageContent: "poiuytrewqasdfghjklm",
        metadata: { b: 2, c: 9, stuff: "right" },
      },
      {
        pageContent: "mnbvcxzasdfghjkloiuy",
        metadata: { b: 1, c: 9, stuff: "right" },
      },
      {
        pageContent: "alskdjfhvngbthrycisudj",
        metadata: { b: 1, c: 9, stuff: "wrong" },
      },
      {
        pageContent: "eeeeeeeeeeeeeeeeeeee",
        metadata: { b: 2, c: 8, stuff: "right" },
      },
      {
        pageContent: "ffffffffffffffffffff",
        metadata: { b: 3, c: 7, stuff: "right" },
      },
      {
        pageContent: "gggggggggggggggggggg",
        metadata: { b: 4, c: 6, stuff: "right" },
      },
    ];

    const getPageContent = (docs: Document<Record<string, any>>[]) =>
      docs.map((doc) => doc.pageContent);
    await pgVS.addDocuments(docs);
    const resultsA = await pgVS.similaritySearch("aaaaaaaaaaaaaaaaaaaa", 2);
    expect(getPageContent(resultsA)).toContain("aaaaaaaaaaaaaaaaaaaa");
    const resultsB = await pgVS.similaritySearch("poiuytrewqasdfghjklm", 2);
    expect(getPageContent(resultsB)).toContain("poiuytrewqasdfghjklm");
    const resultsC = await pgVS.similaritySearch("mnbvcxzasdfghjkloiuy", 2);
    expect(getPageContent(resultsC)).toContain("mnbvcxzasdfghjkloiuy");
    const resultsD = await pgVS.similaritySearch("alskdjfhvngbthrycisudj", 2);
    expect(getPageContent(resultsD)).toContain("alskdjfhvngbthrycisudj");
  });
});

describe("injection tests", () => {
  test("SQL Injection Test 1", async () => {
    const embedding = new FakeEmbeddings();
    // create some_other_stuff table with pg-promise
    await pgvsPgvector.none(
      "CREATE TABLE IF NOT EXISTS injection_some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
    );
    // add some data to some_other_stuff table.
    await pgvsPgvector.none(
      "INSERT INTO injection_some_extra_stuff (type) VALUES ('its'), ('me'), ('hi'), ('im'), ('the'), ('problem'), ('its'), ('me')"
    );

    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "injection_test",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [
        {
          name: "injection_some_extra_column",
          type: "integer",
          returned: true,
          references: { table: "injection_some_extra_stuff", column: "id" },
        },
      ],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    await pgVS.ensureTableInDatabase();

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent: "This is a long text",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "This is a long text",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: 1 },
        { extra_stuff: 2 },
        { extra_stuff: 3 },
      ],
    });

    try {
      await pgVS.maxMarginalRelevanceSearch("hello", {
        k: 3,
        fetchK: 3,
        filter: {
          columnFilter: {
            "extra_stuff = 'weeewooo'); DROP TABLE injection_test; --": {
              $eq: "hi",
            },
          },
        },
      });

      const result = await pgvsPgvector.any(
        "SELECT * FROM information_schema.tables WHERE table_name = $1;",
        ["injection_test"]
      );
      if (result.length === 0) {
        expect("This is bad").toBe("Injection should have failed");
      } else {
        expect(true).toBe(true);
      }
    } catch (error) {
      expect(true).toBe(true);
    }
  });

  test("SQL Injection Test 2", async () => {
    const embedding = new FakeEmbeddings();
    // create some_other_stuff table with pg-promise
    await pgvsPgvector.none(
      "CREATE TABLE IF NOT EXISTS injection_some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
    );
    // add some data to some_other_stuff table.
    await pgvsPgvector.none(
      "INSERT INTO injection_some_extra_stuff (type) VALUES ('its'), ('me'), ('hi'), ('im'), ('the'), ('problem'), ('its'), ('me')"
    );

    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "injection_test",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [
        {
          name: "injection_some_extra_column",
          type: "integer",
          returned: true,
          references: { table: "injection_some_extra_stuff", column: "id" },
        },
      ],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    await pgVS.ensureTableInDatabase();

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent: "This is a long text",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "This is a long text",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: 1 },
        { extra_stuff: 2 },
        { extra_stuff: 3 },
      ],
    });

    try {
      await pgVS.maxMarginalRelevanceSearch("hello", {
        k: 3,
        fetchK: 3,
        filter: {
          metadataFilter: {
            "'b' = 'weeewooo'); DROP TABLE injection_test; --": { $eq: "hi" },
          },
        },
      });

      const result = await pgvsPgvector.any(
        "SELECT * FROM information_schema.tables WHERE table_name = $1;",
        ["injection_test"]
      );
      if (result.length === 0) {
        expect("This is bad").toBe("Injection should have failed");
      } else {
        expect(true).toBe(true);
      }
    } catch (error) {
      expect(true).toBe(true);
    }
  });

  test("SQL Injection Test 3", async () => {
    const embedding = new FakeEmbeddings();
    // create some_other_stuff table with pg-promise
    await pgvsPgvector.none(
      "CREATE TABLE IF NOT EXISTS injection_some_extra_stuff (id serial PRIMARY KEY, type varchar(16))"
    );
    // add some data to some_other_stuff table.
    await pgvsPgvector.none(
      "INSERT INTO injection_some_extra_stuff (type) VALUES ('its'), ('me'), ('hi'), ('im'), ('the'), ('problem'), ('its'), ('me')"
    );

    const pgVS = new PGVectorStore(embedding, {
      postgresConnectionOptions: pgvsPgvector,
      useHnswIndex: true,
      tableName: "injection_test",
      columns: {
        contentColumnName: "content",
      },
      extraColumns: [
        {
          name: "injection_some_extra_column",
          type: "integer",
          returned: true,
          references: { table: "injection_some_extra_stuff", column: "id" },
        },
      ],
      pgExtensionOpts: { type: "pgvector", dims: 4 },
    });

    await pgVS.ensureTableInDatabase();

    const createdAt = new Date().getTime();

    const docs = [
      {
        pageContent: "This is a long text",
        metadata: { b: 1, c: 10, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "This is a long text",
        metadata: { b: 2, c: 9, stuff: "right", created_at: createdAt },
      },
      {
        pageContent: "hello",
        metadata: { b: 1, c: 9, stuff: "right", created_at: createdAt },
      },
    ];

    await pgVS.addDocuments(docs, {
      extraColumns: [
        { extra_stuff: 1 },
        { extra_stuff: 2 },
        { extra_stuff: 3 },
      ],
    });

    try {
      await pgVS.maxMarginalRelevanceSearch("hello", {
        k: 3,
        fetchK: 3,
        filter: {
          join: {
            op: "JOIN",
            table: "injection_some_extra_stuff",
            on: [
              {
                left: "injection_test.injection_some_extra_column",
                right:
                  "injection_some_extra_stuff.id; DROP TABLE injection_test; --",
                operator: "=",
              },
            ],
          },
        },
      });

      const result = await pgvsPgvector.any(
        "SELECT * FROM information_schema.tables WHERE table_name = $1;",
        ["injection_test"]
      );
      if (result.length === 0) {
        expect("This is bad").toBe("Injection should have failed");
      } else {
        expect(true).toBe(true);
      }
    } catch (error) {
      expect(true).toBe(true);
    }
  });
});
