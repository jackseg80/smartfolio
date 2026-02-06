"""
Export Formatter Service - Unified export system for Crypto, Saxo, and Wealth modules.

Supports multiple formats: JSON, CSV, Markdown

Usage:
    from services.export_formatter import ExportFormatter

    # Crypto export
    formatter = ExportFormatter('crypto')
    result = formatter.to_csv(data)

    # Saxo export
    formatter = ExportFormatter('saxo')
    result = formatter.to_markdown(data)

    # Banks export
    formatter = ExportFormatter('banks')
    result = formatter.to_json(data)
"""
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Literal

logger = logging.getLogger(__name__)

ModuleType = Literal['crypto', 'saxo', 'banks', 'wealth']


class ExportFormatter:
    """
    Unified export formatter for all wealth modules.

    Attributes:
        module: Module type (crypto, saxo, banks)
    """

    def __init__(self, module: ModuleType):
        """
        Initialize ExportFormatter.

        Args:
            module: Module type (crypto, saxo, banks)
        """
        self.module = module
        self.timestamp = datetime.utcnow().isoformat() + "Z"

    def to_json(self, data: Dict[str, Any], pretty: bool = True) -> str:
        """
        Convert data to JSON format.

        Args:
            data: Export data dict
            pretty: Pretty print with indentation

        Returns:
            JSON string
        """
        export_data = {
            "module": self.module,
            "exported_at": self.timestamp,
            "data": data
        }

        if pretty:
            return json.dumps(export_data, indent=2, ensure_ascii=False)
        return json.dumps(export_data, ensure_ascii=False)

    def to_csv(self, data: Dict[str, Any]) -> str:
        """
        Convert data to CSV format.

        Args:
            data: Export data dict with 'items' or 'positions' or 'accounts'

        Returns:
            CSV string
        """
        if self.module == 'crypto':
            return self._crypto_to_csv(data)
        elif self.module == 'saxo':
            return self._saxo_to_csv(data)
        elif self.module == 'banks':
            return self._banks_to_csv(data)
        elif self.module == 'wealth':
            return self._wealth_to_csv(data)
        else:
            raise ValueError(f"Unknown module: {self.module}")

    def to_markdown(self, data: Dict[str, Any]) -> str:
        """
        Convert data to Markdown format.

        Args:
            data: Export data dict

        Returns:
            Markdown string
        """
        if self.module == 'crypto':
            return self._crypto_to_markdown(data)
        elif self.module == 'saxo':
            return self._saxo_to_markdown(data)
        elif self.module == 'banks':
            return self._banks_to_markdown(data)
        elif self.module == 'wealth':
            return self._wealth_to_markdown(data)
        else:
            raise ValueError(f"Unknown module: {self.module}")

    # ===== CRYPTO FORMATTERS =====

    def _crypto_to_csv(self, data: Dict[str, Any]) -> str:
        """Format crypto data as CSV."""
        lines = []

        # Header
        lines.append(f"# Crypto Portfolio Export - {self.timestamp}")
        lines.append("")

        # Assets section
        lines.append("Symbol,Group,Amount,Value USD,Location")
        items = data.get('items', [])
        for item in items:
            symbol = item.get('symbol', '')
            group = item.get('group', 'Others')
            amount = item.get('amount', 0)
            value_usd = item.get('value_usd', 0)
            location = item.get('location', '')
            lines.append(f"{symbol},{group},{amount:.8f},{value_usd:.2f},{location}")

        lines.append("")
        lines.append("")

        # Groups section
        lines.append("Group Name,Symbols Count,Portfolio Value USD,Portfolio Percentage")
        groups = data.get('groups', [])
        for group in groups:
            name = group.get('name', '')
            symbols_count = len(group.get('symbols', []))
            total_usd = group.get('portfolio_total_usd', 0)
            percentage = group.get('portfolio_percentage', 0)
            lines.append(f"{name},{symbols_count},{total_usd:.2f},{percentage:.2f}%")

        return "\n".join(lines)

    def _crypto_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format crypto data as Markdown."""
        lines = []

        # Header
        lines.append(f"# 💰 Crypto Portfolio Export")
        lines.append(f"")
        lines.append(f"**Exported:** {self.timestamp}")
        lines.append(f"")

        # Summary
        total_value = sum(item.get('value_usd', 0) for item in data.get('items', []))
        lines.append(f"**Total Portfolio Value:** ${total_value:,.2f}")
        lines.append(f"**Assets Count:** {len(data.get('items', []))}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Assets section
        lines.append(f"## 📊 Assets")
        lines.append(f"")
        lines.append("| Symbol | Group | Amount | Value USD | Location |")
        lines.append("|--------|-------|--------|-----------|----------|")

        items = data.get('items', [])
        for item in items:
            symbol = item.get('symbol', '')
            group = item.get('group', 'Others')
            amount = item.get('amount', 0)
            value_usd = item.get('value_usd', 0)
            location = item.get('location', '')
            lines.append(f"| {symbol} | {group} | {amount:.8f} | ${value_usd:,.2f} | {location} |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Groups section
        lines.append(f"## 🗂️ Groups (11 Categories)")
        lines.append(f"")
        lines.append("| Group | Symbols | Value USD | Allocation % |")
        lines.append("|-------|---------|-----------|--------------|")

        groups = data.get('groups', [])
        for group in groups:
            name = group.get('name', '')
            symbols_list = ", ".join(group.get('symbols', []))
            total_usd = group.get('portfolio_total_usd', 0)
            percentage = group.get('portfolio_percentage', 0)
            lines.append(f"| **{name}** | {symbols_list[:50]}... | ${total_usd:,.2f} | {percentage:.2f}% |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Generated by Crypto Rebalancer - Export System*")

        return "\n".join(lines)

    # ===== SAXO FORMATTERS =====

    def _saxo_to_csv(self, data: Dict[str, Any]) -> str:
        """Format Saxo data as CSV."""
        lines = []

        # Header
        lines.append(f"# Saxo Bank Portfolio Export - {self.timestamp}")
        lines.append("")

        # Positions section
        lines.append("Symbol,Instrument,Asset Class,Quantity,Market Value,Currency,Sector,Entry Price")
        positions = data.get('positions', [])
        for pos in positions:
            symbol = pos.get('symbol', '')
            instrument = pos.get('instrument', '')
            asset_class = pos.get('asset_class', '')
            quantity = pos.get('quantity', 0)
            market_value = pos.get('market_value', 0)
            currency = pos.get('currency', 'USD')
            sector = pos.get('sector', 'Unknown')
            entry_price = pos.get('entry_price', 0)
            lines.append(f"{symbol},{instrument},{asset_class},{quantity:.4f},{market_value:.2f},{currency},{sector},{entry_price:.4f}")

        lines.append("")
        lines.append("")

        # Sectors section
        lines.append("Sector,Value USD,Percentage,Asset Count")
        sectors = data.get('sectors', [])
        for sector in sectors:
            name = sector.get('name', '')
            value_usd = sector.get('value_usd', 0)
            percentage = sector.get('percentage', 0)
            count = sector.get('asset_count', 0)
            lines.append(f"{name},{value_usd:.2f},{percentage:.2f}%,{count}")

        return "\n".join(lines)

    def _saxo_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format Saxo data as Markdown."""
        lines = []

        # Header
        lines.append(f"# 📈 Saxo Bank Portfolio Export")
        lines.append(f"")
        lines.append(f"**Exported:** {self.timestamp}")
        lines.append(f"")

        # Summary
        total_value = sum(pos.get('market_value', 0) for pos in data.get('positions', []))
        lines.append(f"**Total Portfolio Value:** ${total_value:,.2f}")
        lines.append(f"**Positions Count:** {len(data.get('positions', []))}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Positions section
        lines.append(f"## 📊 Positions")
        lines.append(f"")
        lines.append("| Symbol | Instrument | Asset Class | Quantity | Market Value | Currency | Sector |")
        lines.append("|--------|------------|-------------|----------|--------------|----------|--------|")

        positions = data.get('positions', [])
        for pos in positions:
            symbol = pos.get('symbol', '')
            instrument = pos.get('instrument', '')[:30]
            asset_class = pos.get('asset_class', '')
            quantity = pos.get('quantity', 0)
            market_value = pos.get('market_value', 0)
            currency = pos.get('currency', 'USD')
            sector = pos.get('sector', 'Unknown')
            lines.append(f"| {symbol} | {instrument} | {asset_class} | {quantity:.2f} | ${market_value:,.2f} | {currency} | {sector} |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Sectors section
        lines.append(f"## 🗂️ Sectors (GICS Classification)")
        lines.append(f"")
        lines.append("| Sector | Value USD | Allocation % | Assets |")
        lines.append("|--------|-----------|--------------|--------|")

        sectors = data.get('sectors', [])
        for sector in sectors:
            name = sector.get('name', '')
            value_usd = sector.get('value_usd', 0)
            percentage = sector.get('percentage', 0)
            count = sector.get('asset_count', 0)
            lines.append(f"| **{name}** | ${value_usd:,.2f} | {percentage:.2f}% | {count} |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Generated by Crypto Rebalancer - Export System*")

        return "\n".join(lines)

    # ===== BANKS FORMATTERS =====

    def _banks_to_csv(self, data: Dict[str, Any]) -> str:
        """Format banks data as CSV."""
        lines = []

        # Header
        lines.append(f"# Bank Accounts Export - {self.timestamp}")
        lines.append("")

        # Accounts section
        lines.append("Bank Name,Account Type,Balance,Currency,Balance USD")
        accounts = data.get('accounts', [])
        for acc in accounts:
            bank_name = acc.get('bank_name', '')
            account_type = acc.get('account_type', '')
            balance = acc.get('balance', 0)
            currency = acc.get('currency', 'USD')
            balance_usd = acc.get('balance_usd', 0)
            lines.append(f"{bank_name},{account_type},{balance:.2f},{currency},{balance_usd:.2f}")

        lines.append("")
        lines.append("")

        # Summary
        total_usd = sum(acc.get('balance_usd', 0) for acc in accounts)
        lines.append(f"Total Balance USD,{total_usd:.2f}")

        return "\n".join(lines)

    def _banks_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format banks data as Markdown."""
        lines = []

        # Header
        lines.append(f"# 🏦 Bank Accounts Export")
        lines.append(f"")
        lines.append(f"**Exported:** {self.timestamp}")
        lines.append(f"")

        # Summary
        accounts = data.get('accounts', [])
        total_usd = sum(acc.get('balance_usd', 0) for acc in accounts)
        lines.append(f"**Total Balance:** ${total_usd:,.2f}")
        lines.append(f"**Accounts Count:** {len(accounts)}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Accounts section
        lines.append(f"## 💳 Accounts")
        lines.append(f"")
        lines.append("| Bank Name | Account Type | Balance | Currency | Balance USD |")
        lines.append("|-----------|--------------|---------|----------|-------------|")

        for acc in accounts:
            bank_name = acc.get('bank_name', '')
            account_type = acc.get('account_type', '')
            balance = acc.get('balance', 0)
            currency = acc.get('currency', 'USD')
            balance_usd = acc.get('balance_usd', 0)
            lines.append(f"| {bank_name} | {account_type} | {balance:,.2f} | {currency} | ${balance_usd:,.2f} |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Generated by Crypto Rebalancer - Export System*")

        return "\n".join(lines)

    # ===== PATRIMOINE FORMATTERS =====

    def _wealth_to_csv(self, data: Dict[str, Any]) -> str:
        """Format wealth data as CSV."""
        lines = []

        # Header
        lines.append(f"# Patrimoine Export - {self.timestamp}")
        lines.append("")

        summary = data.get('summary', {})
        items_by_category = data.get('items_by_category', {})

        # Summary section
        lines.append("# Summary")
        lines.append(f"Net Worth USD,{summary.get('net_worth', 0):.2f}")
        lines.append(f"Total Assets USD,{summary.get('total_assets', 0):.2f}")
        lines.append(f"Total Liabilities USD,{summary.get('total_liabilities', 0):.2f}")
        lines.append("")

        # Breakdown by category
        breakdown = summary.get('breakdown', {})
        counts = summary.get('counts', {})
        lines.append("# Breakdown by Category")
        lines.append("Category,Total USD,Count")
        lines.append(f"Liquidity,{breakdown.get('liquidity', 0):.2f},{counts.get('liquidity', 0)}")
        lines.append(f"Tangible Assets,{breakdown.get('tangible', 0):.2f},{counts.get('tangible', 0)}")
        lines.append(f"Insurance,{breakdown.get('insurance', 0):.2f},{counts.get('insurance', 0)}")
        lines.append(f"Liabilities,{breakdown.get('liability', 0):.2f},{counts.get('liability', 0)}")
        lines.append("")

        # Items by category
        category_labels = {
            'liquidity': 'Liquidités',
            'tangible': 'Biens Tangibles',
            'liability': 'Passifs',
            'insurance': 'Assurances'
        }

        for category, label in category_labels.items():
            items = items_by_category.get(category, [])
            if items:
                lines.append(f"# {label}")
                lines.append("ID,Name,Type,Value,Currency,Value USD,Acquisition Date,Notes")
                for item in items:
                    lines.append(
                        f"{item.get('id', '')},"
                        f"{item.get('name', '')},"
                        f"{item.get('type', '')},"
                        f"{item.get('value', 0):.2f},"
                        f"{item.get('currency', 'USD')},"
                        f"{item.get('value_usd', 0):.2f},"
                        f"{item.get('acquisition_date', '')},"
                        f"\"{item.get('notes', '')}\""
                    )
                lines.append("")

        return "\n".join(lines)

    def _wealth_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format wealth data as Markdown."""
        lines = []

        # Header
        lines.append(f"# 💰 Patrimoine Export")
        lines.append(f"")
        lines.append(f"**Exported:** {self.timestamp}")
        lines.append(f"")

        # Summary
        summary = data.get('summary', {})
        net_worth = summary.get('net_worth', 0)
        total_assets = summary.get('total_assets', 0)
        total_liabilities = summary.get('total_liabilities', 0)

        lines.append(f"## 📊 Summary")
        lines.append(f"")
        lines.append(f"- **Net Worth:** ${net_worth:,.2f}")
        lines.append(f"- **Total Assets:** ${total_assets:,.2f}")
        lines.append(f"- **Total Liabilities:** ${total_liabilities:,.2f}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Breakdown by category
        breakdown = summary.get('breakdown', {})
        counts = summary.get('counts', {})
        lines.append(f"## 📈 Breakdown by Category")
        lines.append(f"")
        lines.append("| Category | Total USD | Count |")
        lines.append("|----------|-----------|-------|")
        lines.append(f"| 💵 Liquidity | ${breakdown.get('liquidity', 0):,.2f} | {counts.get('liquidity', 0)} |")
        lines.append(f"| 🏠 Tangible Assets | ${breakdown.get('tangible', 0):,.2f} | {counts.get('tangible', 0)} |")
        lines.append(f"| 🛡️ Insurance | ${breakdown.get('insurance', 0):,.2f} | {counts.get('insurance', 0)} |")
        lines.append(f"| ⚠️ Liabilities | ${breakdown.get('liability', 0):,.2f} | {counts.get('liability', 0)} |")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Items by category
        items_by_category = data.get('items_by_category', {})
        category_labels = {
            'liquidity': ('💵', 'Liquidités'),
            'tangible': ('🏠', 'Biens Tangibles'),
            'insurance': ('🛡️', 'Assurances'),
            'liability': ('⚠️', 'Passifs')
        }

        for category, (emoji, label) in category_labels.items():
            items = items_by_category.get(category, [])
            if items:
                lines.append(f"## {emoji} {label}")
                lines.append(f"")
                lines.append("| Name | Type | Value | Currency | Value USD | Acquisition Date | Notes |")
                lines.append("|------|------|-------|----------|-----------|------------------|-------|")
                for item in items:
                    name = item.get('name', '')
                    type_val = item.get('type', '')
                    value = item.get('value', 0)
                    currency = item.get('currency', 'USD')
                    value_usd = item.get('value_usd', 0)
                    acq_date = item.get('acquisition_date', '')
                    notes = item.get('notes', '')[:50]  # Truncate notes
                    lines.append(f"| {name} | {type_val} | {value:,.2f} | {currency} | ${value_usd:,.2f} | {acq_date} | {notes} |")
                lines.append(f"")

        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Generated by Crypto Rebalancer - Export System*")

        return "\n".join(lines)
