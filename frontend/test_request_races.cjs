const test = require('node:test');
const assert = require('node:assert/strict');
const path = require('node:path');
const Module = require('node:module');
const esbuild = require('esbuild');
const React = require('react');
const Renderer = require('react-test-renderer');
const { act } = Renderer;
const parentModule = module;

// Exercise actual component state/effects, mocking child UI and market transport.
async function loadComponent(entry) {
    const result = await esbuild.build({
        entryPoints: [path.join(__dirname, 'src', entry)], bundle: true,
        write: false, platform: 'node', format: 'cjs', external: ['react'],
        plugins: [{ name: 'fixtures', setup(build) {
            function fixture(filter, contents) {
                build.onResolve({filter}, args => ({path:args.path,namespace:'fixture'}));
                build.onLoad({filter,namespace:'fixture'}, args => ({contents:typeof contents==='function'?contents(args.path):contents,resolveDir:__dirname}));
            }
            fixture(/^axios$/, 'export default {post:(...args)=>globalThis.testAxios.post(...args)};');
            fixture(/(^|\/)api$/, 'export const API_BASE_URL="http://test";');
            fixture(/lucide-react/, 'export const Moon=()=>null,Sun=()=>null,TrendingUp=()=>null,DollarSign=()=>null,Activity=()=>null;');
            fixture(/context\/ThemeContext$/, 'export const useTheme=()=>({isDark:false,toggleTheme:()=>{}});');
            fixture(/context\/UserPreferencesContext$/, 'export const UserPreferencesProvider=({children})=>children; export const useUserPreferences=()=>({preferences:{},watchlist:[],updatePreference:()=>{},updateWatchlist:()=>{}});');
            fixture(/components\/(Dashboard|Sidebar|StockScreener|JSEHeatmap|StockIdeasFeed|Watchlist|PortfolioTracker|ConsentToast)$/, p=>`import React from 'react'; export default props=>React.createElement('mock-${p.split('/').pop()}',props);`);
            fixture(/InfoTip$/, 'export default ({children})=>children;');
        }}],
    });
    const compiled = new Module(path.join(__dirname,'test-component.cjs'),parentModule);
    compiled.filename=path.join(__dirname,'test-component.cjs');
    compiled.paths=Module._nodeModulePaths(__dirname);
    compiled._compile(result.outputFiles[0].text,compiled.filename);
    return compiled.exports.default;
}
let App,ValuationLab,pending,trees;
test.before(async()=>{App=await loadComponent('App.jsx');ValuationLab=await loadComponent('components/ValuationLab.jsx');});
test.beforeEach(()=>{
    pending=[];trees=[];
    global.window={location:{hash:'#/'},addEventListener(){},removeEventListener(){},setTimeout};
    global.localStorage={getItem:()=>null,setItem(){}};
    // Ignore abort deliberately: stale-response guards must work independently.
    global.testAxios={post(url,payload,config){return new Promise((resolve,reject)=>pending.push({url,payload,config,resolve,reject,done:false}));}};
});
test.afterEach(()=>{act(()=>trees.forEach(tree=>tree.unmount()));});
async function mount(Component,props){let tree;await act(async()=>{tree=Renderer.create(React.createElement(Component,props));});trees.push(tree);return tree;}
const dashboard=tree=>tree.root.findByType('mock-Dashboard').props;
const requestsFor=ticker=>pending.filter(r=>r.payload.ticker===ticker);
async function select(tree,ticker){await act(async()=>{dashboard(tree).onSelectTicker(ticker);});}
async function settle(requests,fail=false,data=null){await act(async()=>{for(const r of requests.filter(r=>!r.done)){r.done=true;if(fail)r.reject(Error('Delayed failure'));else r.resolve({data:data||{ticker:r.payload.ticker}});}});}

test('late analysis success cannot overwrite newer ticker or fundamentals',async()=>{
    const tree=await mount(App);await select(tree,'AAA.JO');const old=requestsFor('AAA.JO');
    await select(tree,'BBB.JO');assert.ok(old.every(r=>r.config.signal.aborted));
    await settle(requestsFor('BBB.JO'));await settle(old);
    assert.equal(dashboard(tree).data.ticker,'BBB.JO');assert.equal(dashboard(tree).fundamentals.ticker,'BBB.JO');assert.equal(dashboard(tree).loading,false);
});
test('late analysis errors and finally cannot clear newer loading state',async()=>{
    const tree=await mount(App);await select(tree,'AAA.JO');await select(tree,'BBB.JO');
    await settle(requestsFor('AAA.JO'),true);assert.equal(dashboard(tree).loading,true);assert.equal(dashboard(tree).error,null);
    await settle(requestsFor('BBB.JO'));assert.equal(dashboard(tree).loading,false);
});
test('old terminal refresh cannot overwrite a newer analysis',async()=>{
    const tree=await mount(App);await select(tree,'AAA.JO');await settle(requestsFor('AAA.JO'));
    await act(async()=>{dashboard(tree).onRefreshTerminal();});const refresh=pending.filter(r=>!r.done);
    await select(tree,'BBB.JO');assert.ok(refresh.every(r=>r.config.signal.aborted));
    await settle(requestsFor('BBB.JO'));await settle(refresh);assert.equal(dashboard(tree).fundamentals.ticker,'BBB.JO');assert.equal(dashboard(tree).terminalRefreshing,false);
});
test('unmount aborts pending analysis requests',async()=>{
    const tree=await mount(App);await select(tree,'AAA.JO');act(()=>tree.unmount());
    assert.ok(pending.every(r=>r.config.signal.aborted));await settle(pending);
});
test('late valuation success cannot overwrite newer ticker price',async()=>{
    const tree=await mount(ValuationLab,{ticker:'AAA.JO'});
    await act(async()=>{tree.update(React.createElement(ValuationLab,{ticker:'BBB.JO'}));});
    assert.ok(requestsFor('AAA.JO')[0].config.signal.aborted);
    await settle(requestsFor('BBB.JO'),false,{fcf:10000,shares_outstanding:100,price:222,currency:'ZAR'});
    await settle(requestsFor('AAA.JO'),false,{fcf:10000,shares_outstanding:100,price:111,currency:'ZAR'});
    const output=JSON.stringify(tree.toJSON());assert.ok(output.includes('222'));assert.ok(!output.includes('111'));
});
test('failed valuation for new ticker clears the old company data',async t=>{
    t.mock.method(console,'error',()=>{});const tree=await mount(ValuationLab,{ticker:'AAA.JO'});
    await settle(requestsFor('AAA.JO'),false,{fcf:10000,shares_outstanding:100,price:111,currency:'ZAR'});
    await act(async()=>{tree.update(React.createElement(ValuationLab,{ticker:'BBB.JO'}));});await settle(requestsFor('BBB.JO'),true);
    const output=JSON.stringify(tree.toJSON());assert.ok(output.includes('Valuation data unavailable for '));assert.ok(output.includes('BBB.JO'));assert.ok(!output.includes('111'));
});

test('late valuation failure cannot stop loading for the current ticker', async t => {
    t.mock.method(console, 'error', () => {});
    const tree = await mount(ValuationLab, {ticker: 'AAA.JO'});
    await act(async () => {
        tree.update(React.createElement(ValuationLab, {ticker: 'BBB.JO'}));
    });
    await settle(requestsFor('AAA.JO'), true);
    assert.ok(JSON.stringify(tree.toJSON()).includes('Loading Valuation Data'));
    await settle(requestsFor('BBB.JO'), false, {
        fcf: 10000, shares_outstanding: 100, price: 222, currency: 'ZAR',
    });
    assert.ok(JSON.stringify(tree.toJSON()).includes('222'));
});

test('unmount aborts a pending valuation request', async () => {
    const tree = await mount(ValuationLab, {ticker: 'AAA.JO'});
    act(() => tree.unmount());
    assert.ok(pending[0].config.signal.aborted);
    await settle(pending);
});

test('inflation changes replace an in-flight analysis', async () => {
    const tree = await mount(App);
    await select(tree, 'AAA.JO');
    const original = [...pending];
    await act(async () => {
        tree.root.findByType('mock-Sidebar').props.setInflationAdjusted(true);
    });
    assert.ok(original.every(request => request.config.signal.aborted));
    const latest = pending.filter(request => !original.includes(request));
    assert.equal(latest.find(request => request.url.endsWith('/api/analyze')).payload.inflation_rate, .05);
    await settle(latest, false, {ticker: 'AAA.JO', adjusted: true});
    await settle(original, false, {ticker: 'AAA.JO', adjusted: false});
    assert.equal(dashboard(tree).data.adjusted, true);
});
