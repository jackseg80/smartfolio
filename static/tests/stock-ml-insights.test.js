import {jest, test, beforeEach, expect} from '@jest/globals';
const apiCall = jest.fn();
jest.unstable_mockModule('../core/fetcher.js', () => ({apiCall}));
const {loadStockMLInsights} = await import('../components/stock-ml-insights.js');
let scope, root, badge;
const run = () => loadStockMLInsights({root, badge, getScope:()=>scope});
const snapshot = () => ({user_id:scope.user,source:scope.source,market:'stocks',scope:'selected_authenticated_portfolio',portfolio_context:{selected_assets:1},results:[{asset:'TEST',nature:'forecast',target:'future_realized_volatility',horizon:'7d',value:0,availability:'Available',reason:'Evaluated zero',provenance:{}},{asset:'TEST',nature:'forecast',target:'future_realized_volatility',horizon:'30d',value:null,availability:'Unavailable',reason:'Missing artifact',provenance:{}}]});
beforeEach(()=>{apiCall.mockReset();scope={user:'alice',source:'saxobank',fileKey:'selected & file.csv'};document.body.innerHTML='<div id="root"></div><span id="badge"></span>';root=document.getElementById('root');badge=document.getElementById('badge');});
test('selected source/file and partial evaluated zero are rendered', async()=>{
 apiCall.mockResolvedValue({ok:true,data:snapshot()});expect(await run()).toBe(true);
 const [url,opts]=apiCall.mock.calls[0];const q=new URL(url,'http://localhost').searchParams;
 expect(q.get('file_key')).toBe(scope.fileKey);expect(q.get('source')).toBe(scope.source);expect(q.get('mode')).toBe('portfolio');expect(opts.maxRetries).toBe(0);
 expect(root.textContent).toContain('0.00%');expect(root.textContent).toContain('Missing artifact');expect(badge.textContent).toBe('Partial');
});
test.each([[401,'Session expired'],[403,'Access denied'],[0,'timed out'],[503,'HTTP 503']])('HTTP %s clears values and remains retryable',async(status,message)=>{
 apiCall.mockResolvedValueOnce({ok:true,data:snapshot()}).mockResolvedValueOnce({ok:false,status});await run();expect(await run()).toBe(false);
 expect(root.querySelector('table')).toBeNull();expect(root.textContent).toContain(message);expect(badge.textContent).toBe('Unavailable');
 apiCall.mockResolvedValueOnce({ok:true,data:snapshot()});expect(await run()).toBe(true);
});
test.each(['user','source','fileKey'])('late response after %s change is discarded',async(field)=>{
 const data=snapshot();let resolve;apiCall.mockReturnValue(new Promise(r=>{resolve=r;}));const task=run();scope={...scope,[field]:'different'};root.textContent='New selection';resolve({ok:true,data});expect(await task).toBe(false);expect(root.textContent).toBe('New selection');
});
test.each(['user_id','source','market','scope'])('mismatched %s response cannot render',async(field)=>{
 apiCall.mockResolvedValue({ok:true,data:{...snapshot(),[field]:'wrong'}});expect(await run()).toBe(false);expect(root.querySelector('table')).toBeNull();
});
test('unavailable portfolio shows its reason safely and can retry',async()=>{
 apiCall.mockResolvedValue({ok:true,data:{...snapshot(),results:[],portfolio_context:{availability:'Unavailable',reason:'<img src=x> Selected file missing'}}});expect(await run()).toBe(false);expect(root.textContent).toContain('Selected file missing');expect(root.querySelector('img')).toBeNull();
});
