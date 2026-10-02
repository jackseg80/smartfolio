import { jest, test, beforeEach, expect } from '@jest/globals';
jest.unstable_mockModule('../core/auth-guard.js', () => ({getAuthHeaders: () => ({Authorization:'Bearer fixture', 'X-User':localStorage.getItem('activeUser')})}));
await import('../core/bourse-source-context.js');
beforeEach(() => { localStorage.clear(); localStorage.setItem('activeUser','alice'); window.getApiBase=()=>''; global.fetch=jest.fn(); });
test.each(['saxobank_csv','saxobank_api','manual_bourse'])('restores exactly the configured stock source %s by reading only',async source => {
  fetch.mockResolvedValue({ok:true,json:async()=>({data:{category:'bourse',active_source:source,status:'ready'}})});
  expect(await window.readConfiguredBourseSource()).toBe(source);
  expect(fetch).toHaveBeenCalledWith('/api/sources/v2/bourse/active',{headers:{Authorization:'Bearer fixture','X-User':'alice'}});
});
test.each(['not_found','expired','identity_changed'])('does not manufacture a stock source when %s',async failure => {
  fetch.mockImplementation(async()=> { if(failure==='identity_changed') localStorage.setItem('activeUser','bob'); return {ok:failure!=='expired',json:async()=>({data:{category:'bourse',active_source:'saxobank_csv',status:failure==='not_found'?'not_found':'ready'}})}; });
  expect(await window.readConfiguredBourseSource()).toBeNull();
});
