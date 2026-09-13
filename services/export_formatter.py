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
import csv
import io
from datetime import datetime
from typing import Dict, List, Any, Literal

logger = logging.getLogger(__name__)

ModuleType = Literal['crypto', 'saxo', 'banks', 'wealth', 'global']


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
        elif self.module == 'global':
            return self._global_to_csv(data)
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
        elif self.module == 'global':
            return self._global_to_markdown(data)
        else:
            raise ValueError(f"Unknown module: {self.module}")

    # ===== CRYPTO FORMATTERS =====

    def _crypto_to_csv(self, data: Dict[str, Any]) -> str:
        """Format crypto data as CSV."""
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([f"# Crypto Portfolio Export - {self.timestamp}"])
        writer.writerow([])
        writer.writerow(["Symbol", "Group", "Amount", "Value USD", "Location"])
        for item in data.get('items', []):
            writer.writerow([
                item.get('symbol', ''), item.get('group', 'Others'), f"{float(item.get('amount', 0) or 0):.8f}",
                f"{float(item.get('value_usd', 0) or 0):.2f}", item.get('location', ''),
            ])
        writer.writerow([])
        writer.writerow(["Group Name", "Symbols Count", "Portfolio Value USD", "Portfolio Percentage"])
        for group in data.get('groups', []):
            writer.writerow([
                group.get('name', ''), len(group.get('symbols', [])),
                f"{float(group.get('portfolio_total_usd', 0) or 0):.2f}",
                f"{float(group.get('portfolio_percentage', 0) or 0):.2f}%",
            ])
        return output.getvalue().rstrip("\n")

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
        lines.append(f"*Generated by SmartFolio Export System*")

        return "\n".join(lines)

    # ===== SAXO FORMATTERS =====

    def _saxo_to_csv(self, data: Dict[str, Any]) -> str:
        """Format Saxo data as CSV."""
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([f"# Saxo Bank Portfolio Export - {self.timestamp}"])
        writer.writerow([])
        writer.writerow(["Symbol", "Instrument", "Asset Class", "Quantity", "Market Value USD", "Currency", "Classification", "Classification Basis", "Entry Price"])
        for pos in data.get('positions', []):
            writer.writerow([
                pos.get('symbol', ''), pos.get('instrument', ''), pos.get('asset_class', ''),
                f"{float(pos.get('quantity', 0) or 0):.4f}",
                f"{float(pos.get('market_value_usd', pos.get('market_value', 0)) or 0):.2f}",
                pos.get('currency', 'USD'), pos.get('classification', pos.get('sector', 'Unclassified Equity')),
                pos.get('classification_basis', ''), f"{float(pos.get('entry_price', 0) or 0):.4f}",
            ])
        writer.writerow([])
        writer.writerow(["Classification", "Value USD", "Percentage", "Asset Count"])
        for classification in data.get('classifications', data.get('sectors', [])):
            writer.writerow([
                classification.get('name', ''), f"{float(classification.get('value_usd', 0) or 0):.2f}",
                f"{float(classification.get('percentage', 0) or 0):.2f}%", classification.get('asset_count', 0),
            ])
        return output.getvalue().rstrip("\n")

    def _saxo_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format Saxo data as Markdown."""
        lines = []

        # Header
        lines.append(f"# 📈 Saxo Bank Portfolio Export")
        lines.append(f"")
        lines.append(f"**Exported:** {self.timestamp}")
        lines.append(f"")

        # Summary
        total_value = data.get('summary', {}).get('total_value_usd', sum(pos.get('market_value_usd', pos.get('market_value', 0)) for pos in data.get('positions', [])))
        lines.append(f"**Total Portfolio Value:** ${total_value:,.2f}")
        lines.append(f"**Positions Count:** {len(data.get('positions', []))}")
        cash_value = data.get('summary', {}).get('cash_value_usd', 0)
        if cash_value:
            lines.append(f"**Cash:** ${cash_value:,.2f} ({data['summary'].get('cash_amount', 0):,.2f} {data['summary'].get('cash_currency', 'USD')})")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Positions section
        lines.append(f"## 📊 Positions")
        lines.append(f"")
        lines.append("| Symbol | Instrument | Asset Class | Quantity | Market Value USD | Currency | Classification |")
        lines.append("|--------|------------|-------------|----------|------------------|----------|----------------|")

        positions = data.get('positions', [])
        for pos in positions:
            symbol = pos.get('symbol', '')
            instrument = pos.get('instrument', '')[:30]
            asset_class = pos.get('asset_class', '')
            quantity = pos.get('quantity', 0)
            market_value = pos.get('market_value_usd', pos.get('market_value', 0))
            currency = pos.get('currency', 'USD')
            classification = pos.get('classification', pos.get('sector', 'Unclassified Equity'))
            lines.append(f"| {symbol} | {instrument} | {asset_class} | {quantity:.2f} | ${market_value:,.2f} | {currency} | {classification} |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

        # Sectors section
        lines.append(f"## 🗂️ Classifications (GICS for equities; exposure for funds)")
        lines.append(f"")
        lines.append("| Classification | Value USD | Allocation % | Assets |")
        lines.append("|--------|-----------|--------------|--------|")

        sectors = data.get('classifications', data.get('sectors', []))
        for sector in sectors:
            name = sector.get('name', '')
            value_usd = sector.get('value_usd', 0)
            percentage = sector.get('percentage', 0)
            count = sector.get('asset_count', 0)
            lines.append(f"| **{name}** | ${value_usd:,.2f} | {percentage:.2f}% | {count} |")

        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")
        lines.append(f"*Generated by SmartFolio Export System*")

        return "\n".join(lines)

    # ===== BANKS FORMATTERS =====

    def _banks_to_csv(self, data: Dict[str, Any]) -> str:
        """Format banks data as CSV."""
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        accounts = data.get('accounts', [])
        writer.writerow([f"# Bank Accounts Export - {self.timestamp}"])
        writer.writerow([])
        writer.writerow(["Bank Name", "Account Type", "Balance", "Currency", "Balance USD"])
        for account in accounts:
            writer.writerow([
                account.get('bank_name', ''), account.get('account_type', ''),
                f"{float(account.get('balance', 0) or 0):.2f}", account.get('currency', 'USD'),
                f"{float(account.get('balance_usd', 0) or 0):.2f}",
            ])
        writer.writerow([])
        writer.writerow(["Total Balance USD", f"{sum(float(account.get('balance_usd', 0) or 0) for account in accounts):.2f}"])
        return output.getvalue().rstrip("\n")

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
        lines.append(f"*Generated by SmartFolio Export System*")

        return "\n".join(lines)

    # ===== PATRIMOINE FORMATTERS =====

    def _wealth_to_csv(self, data: Dict[str, Any]) -> str:
        """Format wealth data as one spreadsheet-safe table plus totals."""
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        summary = data.get('summary', {})
        items_by_category = data.get('items_by_category', {})
        writer.writerow([f"# Wealth Export - {self.timestamp}"])
        writer.writerow([])
        writer.writerow(["Category", "ID", "Name", "Type", "Value", "Currency", "Value USD", "Acquisition Date", "Notes"])
        for category in ("liquidity", "tangible", "insurance", "liability"):
            for item in items_by_category.get(category, []):
                writer.writerow([
                    category.title(), item.get('id', ''), item.get('name', ''), item.get('type', ''),
                    f"{float(item.get('value', 0) or 0):.2f}", item.get('currency', 'USD'),
                    f"{float(item.get('value_usd', 0) or 0):.2f}", item.get('acquisition_date') or '', item.get('notes') or '',
                ])
        writer.writerow([])
        writer.writerow(["Summary", "Value USD"])
        writer.writerow(["Net Worth", f"{float(summary.get('net_worth', 0) or 0):.2f}"])
        writer.writerow(["Total Assets", f"{float(summary.get('total_assets', 0) or 0):.2f}"])
        writer.writerow(["Total Liabilities", f"{float(summary.get('total_liabilities', 0) or 0):.2f}"])
        return output.getvalue().rstrip("\n")

    def _wealth_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format wealth data as Markdown."""
        lines = []

        # Header
        lines.append(f"# 💰 Wealth Export")
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
            'liquidity': ('💵', 'Liquidities'),
            'tangible': ('🏠', 'Tangible Assets'),
            'insurance': ('🛡️', 'Insurance'),
            'liability': ('⚠️', 'Liabilities')
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
        lines.append(f"*Generated by SmartFolio Export System*")

        return "\n".join(lines)

    # ===== GLOBAL OVERVIEW FORMATTERS =====

    def _global_to_csv(self, data: Dict[str, Any]) -> str:
        """Format all portfolio sources as one spreadsheet-safe table."""
        output = io.StringIO()
        writer = csv.writer(output, lineterminator="\n")
        writer.writerow([f"# Global Overview Export - {self.timestamp}"])
        writer.writerow([])
        writer.writerow(["Source", "Category", "Asset", "Type", "Quantity", "Original Value", "Currency", "Value USD", "Classification", "Notes"])
        for item in data.get("items", []):
            writer.writerow([
                item.get("source", ""), item.get("category", ""), item.get("asset", ""), item.get("type", ""),
                item.get("quantity", ""), item.get("original_value", ""), item.get("currency", ""),
                f"{float(item.get('value_usd', 0) or 0):.2f}", item.get("classification", ""), item.get("notes", ""),
            ])
        writer.writerow([])
        writer.writerow(["Source", "Total USD"])
        for source, value in data.get("summary", {}).get("by_source_usd", {}).items():
            writer.writerow([source, f"{float(value or 0):.2f}"])
        writer.writerow(["Global Total", f"{float(data.get('summary', {}).get('total_value_usd', 0) or 0):.2f}"])
        return output.getvalue().rstrip("\n")

    def _global_to_markdown(self, data: Dict[str, Any]) -> str:
        """Format all portfolio sources as a concise, readable overview."""
        summary = data.get("summary", {})
        lines = [
            "# Global Overview Export",
            "",
            f"**Exported:** {self.timestamp}",
            f"**Global Total:** ${float(summary.get('total_value_usd', 0) or 0):,.2f}",
            "",
            "## Source totals",
            "",
            "| Source | Total USD |",
            "|--------|----------:|",
        ]
        for source, value in summary.get("by_source_usd", {}).items():
            lines.append(f"| {source} | ${float(value or 0):,.2f} |")
        lines.extend([
            "",
            "## Holdings",
            "",
            "| Source | Category | Asset | Type | Value USD | Classification |",
            "|--------|----------|-------|------|----------:|----------------|",
        ])
        for item in data.get("items", []):
            lines.append(
                f"| {item.get('source', '')} | {item.get('category', '')} | {item.get('asset', '')} | "
                f"{item.get('type', '')} | ${float(item.get('value_usd', 0) or 0):,.2f} | {item.get('classification', '')} |"
            )
        return "\n".join(lines)
