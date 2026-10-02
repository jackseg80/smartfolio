import { test, beforeEach, expect } from '@jest/globals';
import { riskRequestScope, riskRequestParams, formatRiskScore } from '../core/risk-request.js';
import { hasAdminNavigation } from '../core/role-policy.js';
import { renderAssetResults, summarizeResults } from '../components/ml-result-table.js';
beforeEach(()=>{localStorage.clear();document.body.replaceChildren();window.globalConfig={get:()=>null};window.userSettings={csv_selected_file:'chosen.csv'};});
test('risk scope isolates user/source/file and aligns windows, threshold and version',()=>{
 localStorage.setItem('activeUser','alice');localStorage.setItem('data_source','csv-a');const a=riskRequestScope();
 expect(riskRequestParams(a)).toEqual({source:'csv-a',min_usd:1,price_history_days:365,lookback_days:90,use_dual_window:true,risk_version:'v2_active',_csv_hint:'chosen.csv'});
 localStorage.setItem('activeUser','bob');localStorage.setItem('data_source','csv-b');expect(riskRequestScope()).not.toEqual(a);
 localStorage.removeItem('activeUser');expect(()=>riskRequestParams()).toThrow('Authenticated');
});
test('risk score preserves zero, rounds values and distinguishes missing',()=>{expect(formatRiskScore(0)).toBe('0/100');expect(formatRiskScore(71.543887565)).toBe('72/100');expect(formatRiskScore(null)).toBe('Unavailable');});
test('Admin visibility follows authenticated identity and admin role only',()=>{expect(hasAdminNavigation({id:'jack',roles:['admin']},'jack')).toBe(true);expect(hasAdminNavigation({id:'jack',roles:['admin']},'alice')).toBe(false);expect(hasAdminNavigation({id:'jack',roles:['ml_admin']},'jack')).toBe(false);expect(hasAdminNavigation(null,'jack')).toBe(false);});
test('one row per asset retains unavailable reasons and zero forecast without inflated coverage',()=>{
 const result=(asset,horizon,value,availability='Available')=>({asset,horizon,value,availability,target:'future_realized_volatility',nature:'forecast',reason:'Missing exact provider history',provenance:{provider:'<img src=x>',dataset_id:'real-receipt'},validation:{state:'retrospectively_validated'}});
 const rows=[result('AAA','7d',0),result('AAA','30d',null,'Unavailable'),result('BBB','7d',null,'Unavailable'),result('BBB','30d',null,'Unavailable')];
 expect(summarizeResults(rows)).toEqual({requested:4,available:1,assets:2,covered:1,diagnostics:0});
 renderAssetResults(document.body,rows);expect(document.querySelectorAll('tbody tr')).toHaveLength(2);expect(document.body.textContent).toContain('0.00%');expect(document.body.textContent).toContain('Missing exact provider history');expect(document.querySelector('img')).toBeNull();expect(document.querySelectorAll('td > details')).toHaveLength(2);
});
