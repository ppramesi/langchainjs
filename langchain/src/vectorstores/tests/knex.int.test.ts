/* eslint-disable no-process-env */
import {
  test,
  expect,
  beforeAll,
  beforeEach,
  afterAll,
  afterEach,
} from "@jest/globals";
import Knex from "knex";
import { OpenAIEmbeddings } from "../../embeddings/openai.js";
import { KnexVectorStore } from "../knex.js";

let knex: Knex.Knex;
if (!process.env.KNEX_POSTGRES_URL) {
  throw new Error("KNEX_POSTGRES_URL not set");
}

beforeAll(async () => {
  knex = Knex.default({
    client: "postgresql",
    connection: {
      connectionString: process.env.KNEX_POSTGRES_URL,
      ssl: {
        rejectUnauthorized: false,
      },
    },
  });
});

beforeEach(async () => {
  await knex.raw(`CREATE EXTENSION IF NOT EXISTS "uuid-ossp";`);
  await knex.raw("CREATE EXTENSION IF NOT EXISTS vector;");

  /**
   * 🚨🚨🚨 WARNING WARNING WARNING 🚨🚨🚨
   * We're dropping knex_embeddings table first to make sure the test
   * is idempotent. This means that if you have a table called
   * knex_embeddings in your database, it will be dropped.
   */
  await knex.schema.dropTableIfExists("knex_embeddings");

  await knex.schema.createTableIfNotExists("knex_embeddings", (table) => {
    table.uuid("id").primary().defaultTo(knex.raw("uuid_generate_v4()"));
    table.text("content");
    table.specificType("embedding", "vector");
    table.jsonb("metadata");
    table.text("extra_stuff");
  });
});

afterEach(async () => {
  await knex.schema.dropTable("knex_embeddings");
});

afterAll(async () => {
  await knex.destroy();
});

test("Build index pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
  });
  try {
    await knexVS.buildIndex();
    expect(true).toBe(true);
  } catch (error) {
    expect(error).toBe(null);
  }
});

test("MMR and Similarity Search Test pgvector", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
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

  await knexVS.addDocuments(docs, {
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

  const ssResults = await knexVS.similaritySearch("hello", 7);

  expect(ssResults.length).toBe(7);

  const mmrResults = await knexVS.maxMarginalRelevanceSearch("hello", {
    k: 3,
    fetchK: 7,
  });

  expect(mmrResults.length).toBe(3);
});

test("Building WHERE query test pgvector", () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
  });

  const queryMetadata = knexVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
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

  expect(queryMetadata).toBe(
    `WHERE (metadata->>"stuff" = 'hello'::text OR metadata->>"hello" = 'stuff'::text OR (to_tsvector('simple', 'content') @@ plainto_tsquery('english', 'hello')))`
  );

  const queryColumn = knexVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
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

  expect(queryColumn).toBe(
    `WHERE ("stuff" = 'hello' OR "hello" = 'stuff' OR (to_tsvector('simple', 'content') @@ plainto_tsquery('english', 'hello')))`
  );
});

// Skipping tests for pgembedding because you need to install pgembedding extension first, so like install pgembedding and then do separate test.
test.skip("Build index pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
  });
  try {
    await knexVS.buildIndex();
    expect(true).toBe(true);
  } catch (error) {
    expect(error).toBe(null);
  }
});

test.skip("MMR and Similarity Search Test pgembedding", async () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
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

  await knexVS.addDocuments(docs, {
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

  const ssResults = await knexVS.similaritySearch("hello", 7);

  expect(ssResults.length).toBe(7);

  const mmrResults = await knexVS.maxMarginalRelevanceSearch("hello", {
    k: 3,
    fetchK: 7,
  });

  expect(mmrResults.length).toBe(3);
});

test.skip("Building WHERE query test pgembedding", () => {
  const embedding = new OpenAIEmbeddings();
  const knexVS = new KnexVectorStore(embedding, {
    knex,
    useHnswIndex: false,
    tableName: "knex_embeddings",
    pageContentColumn: "content",
    extraColumns: [{ name: "extra_stuff", type: "text", returned: true }],
    pgExtension: "pgembedding",
  });

  const queryMetadata = knexVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
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

  expect(queryMetadata).toBe(
    `WHERE (metadata->>"stuff" = 'hello'::text OR metadata->>"hello" = 'stuff'::text OR (to_tsvector('simple', 'content') @@ plainto_tsquery('english', 'hello')))`
  );

  const queryColumn = knexVS.buildSqlFilterStr(
    {
      $or: [
        { stuff: { $eq: "hello" } },
        { hello: "stuff" },
        {
          $and: [
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

  expect(queryColumn).toBe(
    `WHERE ("stuff" = 'hello' OR "hello" = 'stuff' OR (to_tsvector('simple', 'content') @@ plainto_tsquery('english', 'hello')))`
  );
});