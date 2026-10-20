// Exercise the native boundary in subprocesses: a panic must never kill the test runner.
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { spawnSync } = require('node:child_process');
const path = require('node:path');
const binding = process.env.RUVECTOR_GNN_BINDING || path.resolve(__dirname, '../index.js');

function runNative(source) {
  const child = spawnSync(process.execPath, ['-e', `
    const assert = require('node:assert/strict');
    const g = require(${JSON.stringify(binding)});
    const q = new Float32Array([1, 0]);
    const layer = new g.RuvectorLayer(2, 2, 1, 0);
    ${source}
  `], { encoding: 'utf8', timeout: 10000 });
  assert.ifError(child.error);
  assert.equal(child.signal, null, child.stderr);
  assert.equal(child.status, 0, child.stderr);
}

for (const length of [0, 1, 3]) {
  test(`differentiableSearch rejects candidate dimension ${length} without aborting`, () => {
    runNative(`assert.throws(() => g.differentiableSearch(q,
      [q, new Float32Array(${length})], 1, 1), {code: 'InvalidArg'});`);
  });
  test(`forward rejects node dimension ${length} without aborting`, () => {
    runNative(`assert.throws(() => layer.forward(new Float32Array(${length}), [],
      new Float32Array()), {code: 'InvalidArg'});`);
  });
  test(`forward rejects neighbor dimension ${length} without aborting`, () => {
    runNative(`assert.throws(() => layer.forward(q, [new Float32Array(${length})],
      new Float32Array([1])), {code: 'InvalidArg'});`);
  });
  test(`hierarchicalForward rejects candidate dimension ${length} without aborting`, () => {
    runNative(`assert.throws(() => g.hierarchicalForward(q,
      [[q, new Float32Array(${length})]], [layer.toJson()]), {code: 'InvalidArg'});`);
  });
}

test('hierarchicalForward checks query against the layer input dimension', () => {
  runNative(`const wide = new g.RuvectorLayer(3, 2, 1, 0);
    assert.throws(() => g.hierarchicalForward(q, [[q]], [wide.toJson()]),
      {code: 'InvalidArg'});`);
});

test('constructor rejects zero attention heads without aborting', () => {
  runNative(`assert.throws(() => new g.RuvectorLayer(2, 2, 0, 0),
    {code: 'InvalidArg'});`);
});

test('hierarchicalForward checks dimensions after an earlier projection', () => {
  runNative(`const wide = new g.RuvectorLayer(2, 4, 1, 0);
    assert.throws(() => g.hierarchicalForward(q, [[q], [q]],
      [wide.toJson(), layer.toJson()]), {code: 'InvalidArg'});`);
});

test('valid search and projected/serialized layers still return finite outputs', () => {
  runNative(`assert.equal(g.differentiableSearch(q, [q, new Float32Array([0,1])],
      1, 1).indices[0], 0);
    const wide = new g.RuvectorLayer(2, 4, 1, 0);
    const restored = g.RuvectorLayer.fromJson(wide.toJson());
    for (const output of [wide.forward(q, [], new Float32Array()),
      restored.forward(q, [q], new Float32Array([1])),
      g.hierarchicalForward(q, [[q]], [wide.toJson()])]) {
      assert.equal(output.length, 4);
      assert.ok(Array.from(output).every(Number.isFinite));
    }
    const next = new g.RuvectorLayer(4, 2, 1, 0);
    const output = g.hierarchicalForward(q, [[q], [new Float32Array([1,0,0,0])]],
      [wide.toJson(), next.toJson()]);
    assert.equal(output.length, 2);
    assert.ok(Array.from(output).every(Number.isFinite));
    assert.equal(g.differentiableSearch(q, [], 1, 1).indices.length, 0);
    assert.deepEqual(Array.from(g.hierarchicalForward(q, [], [])), Array.from(q));`);
});

test('hierarchicalForward preserves empty and unmatched layer behavior', () => {
  runNative(`assert.deepEqual(Array.from(g.hierarchicalForward(q, [[],[q]],
    [layer.toJson()])), Array.from(q));
    assert.deepEqual(Array.from(g.hierarchicalForward(q, [], [layer.toJson()])), Array.from(q));
    assert.deepEqual(Array.from(g.hierarchicalForward(q, [[q]], [])), Array.from(q));`);
});
