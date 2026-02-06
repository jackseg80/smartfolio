// Script d'initialisation des ancres pour toutes les pages canoniques
// Import automatique selon la page actuelle

const pageAnchors = {
  'dashboard.html': {
    'overview': 'Overview',
    'crypto': 'Crypto Portfolio',
    'bourse': 'Stocks & ETF',
    'banque': 'Bank Accounts',
    'divers': 'Miscellaneous Assets',
    'fx': 'Currencies & Forex'
  },
  'analytics-unified.html': {
    'unified': 'Unified Analytics',
    'ml': 'Machine Learning',
    'cycles': 'Cycle Analysis',
    'performance': 'Performance',
    'monitoring': 'Monitoring'
  },
  'risk-dashboard.html': {
    'governance': 'Governance',
    'stress': 'Stress Tests',
    'risk-attribution': 'Risk Attribution',
    'limits': 'Limits & Controls'
  },
  'rebalance.html': {
    'proposed-targets': 'Proposed Targets',
    'bourse': 'Rebalance Stocks',
    'funding-plan': 'Funding Plan',
    'dca-schedule': 'DCA Schedule',
    'constraints': 'Constraints'
  },
  'execution.html': {
    'orders': 'Orders',
    'history': 'History',
    'costs': 'Costs',
    'venues': 'Platforms'
  },
  'settings.html': {
    'integrations': 'Integrations',
    'security': 'Security',
    'logs': 'Logs',
    'monitoring': 'Monitoring',
    'tools': 'Tools'
  }
};

// Auto-détection de la page et initialisation
const currentPage = location.pathname.split('/').pop() || 'index.html';
const anchors = pageAnchors[currentPage];

if (anchors) {
  import('./deep-links.js').then(({ initDeepLinks }) => {
    initDeepLinks(anchors);
  });
}