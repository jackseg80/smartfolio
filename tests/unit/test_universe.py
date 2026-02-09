# tests/unit/test_universe.py
from __future__ import annotations
import json, math, os, time
from dataclasses import asdict
from unittest.mock import MagicMock, patch
import pytest
from connectors.coingecko import CoinMeta
from services.universe import ScoredCoin, UniverseCache, UniverseManager, get_universe_manager


def _make_coin(symbol="BTC", alias="BTC", coingecko_id="bitcoin", market_cap_rank=1, volume_24h=5e9, price_change_30d=10.0, price_change_90d=25.0, liquidity_proxy=0.12, risk_flags=None):
    return CoinMeta(symbol=symbol, alias=alias, coingecko_id=coingecko_id, market_cap_rank=market_cap_rank, volume_24h=volume_24h, price_change_30d=price_change_30d, price_change_90d=price_change_90d, liquidity_proxy=liquidity_proxy, risk_flags=risk_flags if risk_flags is not None else [])


def _w():
    return {"w_cap_rank_inv": 0.3, "w_liquidity": 0.25, "w_momentum": 0.2, "w_internal": 0.1, "w_risk": 0.15}


def _g():
    return {"min_liquidity_usd": 50000, "max_weight_per_coin": 0.4, "min_trade_usd_default": 25.0}


@pytest.fixture
def manager():
    return UniverseManager(config_path="__nonexistent__.json")


@pytest.fixture
def btc_coin():
    return _make_coin()


@pytest.fixture
def eth_coin():
    return _make_coin(symbol="ETH", alias="ETH", coingecko_id="ethereum", market_cap_rank=2, volume_24h=3e9, price_change_30d=5.0, price_change_90d=15.0, liquidity_proxy=0.10)


class TestScoredCoinSerialization:
    def test_to_dict_keys(self, btc_coin):
        assert set(ScoredCoin(meta=btc_coin, score=0.85, reasons={"a": 0.3}).to_dict().keys()) == {"meta", "score", "reasons"}

    def test_roundtrip(self, btc_coin):
        o = ScoredCoin(meta=btc_coin, score=0.75, reasons={"cap_rank_inv": 0.25})
        r = ScoredCoin.from_dict(o.to_dict())
        assert r.score == o.score and r.reasons == o.reasons and r.meta.symbol == o.meta.symbol

    def test_meta_fields(self, btc_coin):
        r = ScoredCoin.from_dict(ScoredCoin(meta=btc_coin, score=1.0, reasons={}).to_dict())
        assert r.meta.market_cap_rank == btc_coin.market_cap_rank
        assert r.meta.volume_24h == btc_coin.volume_24h
        assert r.meta.liquidity_proxy == btc_coin.liquidity_proxy

    def test_meta_is_dict(self, btc_coin):
        d = ScoredCoin(meta=btc_coin, score=0.5, reasons={}).to_dict()
        assert isinstance(d["meta"], dict) and d["meta"]["symbol"] == "BTC"

    def test_risk_flags(self):
        c = _make_coin(risk_flags=["small_cap", "low_volume"])
        r = ScoredCoin.from_dict(ScoredCoin(meta=c, score=0.3, reasons={}).to_dict())
        assert r.meta.risk_flags == ["small_cap", "low_volume"]

    def test_empty_reasons(self, btc_coin):
        r = ScoredCoin.from_dict(ScoredCoin(meta=btc_coin, score=0.0, reasons={}).to_dict())
        assert r.reasons == {} and r.score == 0.0


class TestUniverseCacheExpiration:
    def test_not_expired(self):
        assert UniverseCache(timestamp=time.time(), last_success_at=time.time(), source="live", ttl_seconds=3600, scored_by_group={}).is_expired() is False

    def test_expired(self):
        assert UniverseCache(timestamp=time.time()-7200, last_success_at=time.time()-7200, source="live", ttl_seconds=3600, scored_by_group={}).is_expired() is True

    def test_zero_ttl(self):
        assert UniverseCache(timestamp=time.time()-0.01, last_success_at=time.time(), source="c", ttl_seconds=0, scored_by_group={}).is_expired() is True

    def test_large_ttl(self):
        assert UniverseCache(timestamp=time.time()-86400, last_success_at=time.time(), source="live", ttl_seconds=86400*365, scored_by_group={}).is_expired() is False

    def test_roundtrip(self, btc_coin):
        sc = ScoredCoin(meta=btc_coin, score=0.9, reasons={"a": 0.3})
        c = UniverseCache(timestamp=1.7e9, last_success_at=1.7e9, source="live", ttl_seconds=3600, scored_by_group={"BTC": [sc]})
        r = UniverseCache.from_dict(c.to_dict())
        assert r.timestamp == c.timestamp and r.source == c.source and r.scored_by_group["BTC"][0].score == 0.9

    def test_empty_groups(self):
        assert UniverseCache.from_dict({"timestamp": 1.7e9, "last_success_at": 1.7e9, "source": "c", "ttl_seconds": 1800, "scored_by_group": {}}).scored_by_group == {}

    def test_missing_key(self):
        assert UniverseCache.from_dict({"timestamp": 1.7e9, "last_success_at": 1.7e9, "source": "c", "ttl_seconds": 1800}).scored_by_group == {}

    def test_multi_groups(self, btc_coin, eth_coin):
        c = UniverseCache(timestamp=1.7e9, last_success_at=1.7e9, source="live", ttl_seconds=3600, scored_by_group={"BTC": [ScoredCoin(meta=btc_coin, score=0.9, reasons={})], "ETH": [ScoredCoin(meta=eth_coin, score=0.8, reasons={})]})
        r = UniverseCache.from_dict(c.to_dict())
        assert set(r.scored_by_group.keys()) == {"BTC", "ETH"} and r.scored_by_group["ETH"][0].meta.symbol == "ETH"


class TestScoreCalculation:
    def test_rank1(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(market_cap_rank=1), _w(), _g())["cap_rank_inv"] - 0.3) < 1e-9

    def test_rank10(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(market_cap_rank=10), _w(), _g())["cap_rank_inv"] - (1.0-math.log10(10)/3.0)*0.3) < 1e-6

    def test_rank100(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(market_cap_rank=100), _w(), _g())["cap_rank_inv"] - (1.0-2.0/3.0)*0.3) < 1e-6

    def test_rank1000(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(market_cap_rank=1000), _w(), _g())["cap_rank_inv"]) < 1e-9

    def test_rank5000(self, manager):
        assert manager._calculate_score_components(_make_coin(market_cap_rank=5000), _w(), _g())["cap_rank_inv"] == 0.0

    def test_rank_none(self, manager):
        assert manager._calculate_score_components(_make_coin(market_cap_rank=None), _w(), _g())["cap_rank_inv"] == 0.0

    def test_liq_proxy(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(liquidity_proxy=0.10), _w(), _g())["liquidity"] - 0.10*2.0*0.25) < 1e-9

    def test_liq_cap(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(liquidity_proxy=0.80), _w(), _g())["liquidity"] - 0.25) < 1e-9

    def test_liq_fb_ok(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(liquidity_proxy=None, volume_24h=500000.0), _w(), _g())["liquidity"] - 0.5*0.25) < 1e-9

    def test_liq_fb_low(self, manager):
        assert manager._calculate_score_components(_make_coin(liquidity_proxy=None, volume_24h=10000.0), _w(), _g())["liquidity"] == 0.0

    def test_liq_none(self, manager):
        assert manager._calculate_score_components(_make_coin(liquidity_proxy=None, volume_24h=None), _w(), _g())["liquidity"] == 0.0

    def test_mom_both(self, manager):
        r = manager._calculate_score_components(_make_coin(price_change_30d=25.0, price_change_90d=50.0), _w(), _g())
        assert abs(r["momentum"] - (max(0,min(1,75/150))*0.6+max(0,min(1,100/150))*0.4)*0.2) < 1e-9

    def test_mom_30d(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(price_change_30d=10.0, price_change_90d=None), _w(), _g())["momentum"] - max(0,min(1,60/150))*0.2) < 1e-9

    def test_mom_neg(self, manager):
        assert manager._calculate_score_components(_make_coin(price_change_30d=-80.0, price_change_90d=-90.0), _w(), _g())["momentum"] == 0.0

    def test_mom_big(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(price_change_30d=200.0, price_change_90d=300.0), _w(), _g())["momentum"] - 0.2) < 1e-9

    def test_mom_none(self, manager):
        assert manager._calculate_score_components(_make_coin(price_change_30d=None, price_change_90d=None), _w(), _g())["momentum"] == 0.0

    def test_internal(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(), _w(), _g())["internal"] - 0.05) < 1e-9

    def test_risk_empty(self, manager):
        assert manager._calculate_score_components(_make_coin(risk_flags=[]), _w(), _g())["risk_penalty"] == 0.0

    def test_risk_sc(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(risk_flags=["small_cap"]), _w(), _g())["risk_penalty"] - (-0.3*0.15)) < 1e-9

    def test_risk_lv(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(risk_flags=["low_volume"]), _w(), _g())["risk_penalty"] - (-0.4*0.15)) < 1e-9

    def test_risk_id(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(risk_flags=["incomplete_data"]), _w(), _g())["risk_penalty"] - (-0.2*0.15)) < 1e-9

    def test_risk_multi(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(risk_flags=["small_cap","low_volume"]), _w(), _g())["risk_penalty"] - (-0.7*0.15)) < 1e-9

    def test_risk_all(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(risk_flags=["small_cap","low_volume","incomplete_data"]), _w(), _g())["risk_penalty"] - (-0.9*0.15)) < 1e-9

    def test_risk_unk(self, manager):
        assert manager._calculate_score_components(_make_coin(risk_flags=["xyz"]), _w(), _g())["risk_penalty"] == 0.0

    def test_keys(self, manager):
        assert set(manager._calculate_score_components(_make_coin(), _w(), _g()).keys()) == {"cap_rank_inv","liquidity","momentum","internal","risk_penalty"}

    def test_custom_w(self, manager):
        r = manager._calculate_score_components(_make_coin(market_cap_rank=1), {"w_cap_rank_inv":1.0,"w_liquidity":0.0,"w_momentum":0.0,"w_internal":0.0,"w_risk":0.0}, _g())
        assert r["cap_rank_inv"] == 1.0 and r["liquidity"] == 0.0

    def test_liq_fb_cap(self, manager):
        assert abs(manager._calculate_score_components(_make_coin(liquidity_proxy=None, volume_24h=2e6), _w(), _g())["liquidity"] - 0.25) < 1e-9


class TestDefaultConfig:
    def test_sections(self, manager):
        c = manager._get_default_config()
        for k in ["features","scoring","allocation","guardrails","lists","cache"]:
            assert k in c

    def test_wsum(self, manager):
        assert abs(sum(manager._get_default_config()["scoring"]["weights"].values()) - 1.0) < 1e-9

    def test_enabled(self, manager):
        assert manager._get_default_config()["features"]["priority_allocation"] is True

    def test_topn(self, manager):
        assert manager._get_default_config()["allocation"]["top_n"] == 3

    def test_decay(self, manager):
        assert abs(sum(manager._get_default_config()["allocation"]["decay"]) - 1.0) < 1e-9

    def test_guard(self, manager):
        g = manager._get_default_config()["guardrails"]
        assert g["min_liquidity_usd"] == 50000 and g["max_weight_per_coin"] == 0.4

    def test_ttl(self, manager):
        assert manager._get_default_config()["cache"]["ttl_seconds"] == 3600

    def test_bl(self, manager):
        assert manager._get_default_config()["lists"]["global_blacklist"] == []

    def test_fallback(self, manager):
        assert manager._load_config() == manager._get_default_config()

    def test_cached(self, manager):
        assert manager._load_config() is manager._load_config()


class TestScoreGroupUniverse:
    def test_sorted(self, manager, btc_coin, eth_coin):
        r = manager.score_group_universe({"g": [btc_coin, eth_coin]})
        s = [sc.score for sc in r["g"]]
        assert s == sorted(s, reverse=True)

    def test_blacklist(self, manager, btc_coin, eth_coin):
        manager._config = manager._get_default_config()
        manager._config["lists"]["global_blacklist"] = ["ETH"]
        r = manager.score_group_universe({"g": [btc_coin, eth_coin]})
        assert all(sc.meta.symbol != "ETH" for sc in r["g"])

    def test_bl_case_sensitive(self, manager, btc_coin, eth_coin):
        manager._config = manager._get_default_config()
        manager._config["lists"]["global_blacklist"] = ["eth"]
        r = manager.score_group_universe({"g": [btc_coin, eth_coin]})
        # Blacklist check uppercases coin.symbol but not blacklist entries
        # So lowercase "eth" does NOT match uppercase "ETH" - coin stays
        assert any(sc.meta.symbol == "ETH" for sc in r["g"])

    def test_empty(self, manager):
        assert manager.score_group_universe({}) == {}

    def test_empty_grp(self, manager):
        assert manager.score_group_universe({"x": []})["x"] == []

    def test_multi(self, manager, btc_coin, eth_coin):
        r = manager.score_group_universe({"B": [btc_coin], "E": [eth_coin]})
        assert len(r["B"]) == 1 and len(r["E"]) == 1

    def test_risky(self, manager):
        g = _make_coin(symbol="G", alias="G")
        b = _make_coin(symbol="B", alias="B", risk_flags=["small_cap","low_volume"])
        r = manager.score_group_universe({"t": [g, b]})
        sc = {x.meta.symbol: x.score for x in r["t"]}
        assert sc["G"] > sc["B"]

    def test_rank(self, manager):
        t = _make_coin(symbol="T", alias="T", market_cap_rank=1)
        l = _make_coin(symbol="L", alias="L", market_cap_rank=500)
        r = manager.score_group_universe({"t": [t, l]})
        assert next(x for x in r["t"] if x.meta.symbol=="T").reasons["cap_rank_inv"] > next(x for x in r["t"] if x.meta.symbol=="L").reasons["cap_rank_inv"]


class TestBuildGroupUniverse:
    @patch("services.universe.get_connector")
    def test_empty(self, mg, manager):
        assert manager.build_group_universe([]) == {}

    @patch("services.universe.get_connector")
    def test_no_port(self, mg, manager):
        assert manager.build_group_universe(["BTC"], current_portfolio=None) == {}

    @patch("services.universe.get_connector")
    def test_with_data(self, mg, manager):
        mc = MagicMock(); mc.get_market_snapshot.return_value = {"BTC": _make_coin()}; mg.return_value = mc
        manager._taxonomy = MagicMock(); manager._taxonomy.get_group.return_value = "BTC"
        assert len(manager.build_group_universe(["BTC"], current_portfolio=[{"symbol":"BTC","alias":"BTC"}])["BTC"]) == 1

    @patch("services.universe.get_connector")
    def test_filter(self, mg, manager):
        mc = MagicMock(); mc.get_market_snapshot.return_value = {"BTC": _make_coin(), "ETH": _make_coin(symbol="ETH",alias="ETH",coingecko_id="ethereum")}; mg.return_value = mc
        manager._taxonomy = MagicMock(); manager._taxonomy.get_group.side_effect = lambda a: "BTC" if a == "BTC" else "ETH"
        r = manager.build_group_universe(["BTC"], current_portfolio=[{"symbol":"BTC","alias":"BTC"},{"symbol":"ETH","alias":"ETH"}])
        assert "BTC" in r and "ETH" not in r

    @patch("services.universe.get_connector")
    def test_no_data(self, mg, manager):
        mc = MagicMock(); mc.get_market_snapshot.return_value = {}; mg.return_value = mc
        manager._taxonomy = MagicMock(); manager._taxonomy.get_group.return_value = "BTC"
        assert manager.build_group_universe(["BTC"], current_portfolio=[{"symbol":"BTC","alias":"BTC"}])["BTC"] == []


class TestGetUniverseCached:
    def test_disabled(self, manager):
        manager._config = manager._get_default_config(); manager._config["features"]["priority_allocation"] = False
        assert manager.get_universe_cached(["BTC"]) is None

    @patch.object(UniverseManager, "_load_cache")
    def test_valid(self, ml, manager):
        manager._config = manager._get_default_config()
        ml.return_value = UniverseCache(timestamp=time.time(), last_success_at=time.time(), source="cache", ttl_seconds=3600, scored_by_group={"BTC": [ScoredCoin(meta=_make_coin(), score=0.9, reasons={})]})
        r = manager.get_universe_cached(["BTC"], mode="prefer_cache")
        assert r is not None and "BTC" in r

    @patch.object(UniverseManager, "_load_cache")
    def test_co_miss(self, ml, manager):
        manager._config = manager._get_default_config(); ml.return_value = None
        assert manager.get_universe_cached(["BTC"], mode="cache_only") is None

    @patch.object(UniverseManager, "_load_cache")
    def test_co_exp(self, ml, manager):
        manager._config = manager._get_default_config()
        ml.return_value = UniverseCache(timestamp=time.time()-7200, last_success_at=time.time()-7200, source="cache", ttl_seconds=3600, scored_by_group={"BTC": []})
        assert manager.get_universe_cached(["BTC"], mode="cache_only") is None

    @patch.object(UniverseManager, "_get_universe_live")
    def test_live(self, ml, manager):
        manager._config = manager._get_default_config(); ml.return_value = {"BTC": []}
        assert manager.get_universe_cached(["BTC"], mode="live_only") == {"BTC": []}

    @patch.object(UniverseManager, "_load_cache")
    def test_miss_grp(self, ml, manager):
        manager._config = manager._get_default_config()
        ml.return_value = UniverseCache(timestamp=time.time(), last_success_at=time.time(), source="cache", ttl_seconds=3600, scored_by_group={"ETH": []})
        with patch.object(manager, "_get_universe_live", return_value={"BTC": []}) as lv:
            manager.get_universe_cached(["BTC"], mode="prefer_cache"); lv.assert_called_once()

    @patch.object(UniverseManager, "_load_cache")
    def test_filt(self, ml, manager):
        manager._config = manager._get_default_config()
        ml.return_value = UniverseCache(timestamp=time.time(), last_success_at=time.time(), source="cache", ttl_seconds=3600, scored_by_group={"BTC": [ScoredCoin(meta=_make_coin(), score=0.9, reasons={})], "ETH": [ScoredCoin(meta=_make_coin(symbol="ETH",alias="ETH",coingecko_id="ethereum"), score=0.8, reasons={})]})
        r = manager.get_universe_cached(["BTC"], mode="prefer_cache")
        assert "BTC" in r and "ETH" not in r


class TestGlobalManager:
    def test_type(self):
        import services.universe as m; m._global_manager = None
        assert isinstance(get_universe_manager(), UniverseManager)

    def test_single(self):
        import services.universe as m; m._global_manager = None
        assert get_universe_manager() is get_universe_manager()
