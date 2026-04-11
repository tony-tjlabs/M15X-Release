"""
Gateway Index - 52개 S-Ward 좌표 기반 최근접 할당
==================================================
M15X 현장은 FAB 내 41개 + 외부 시설 11개 S-Ward로 구성.
작업자 (x, y, floor) 좌표 -> 가장 가까운 S-Ward 할당.

사용법:
    from src.spatial.gateway_index import GatewayIndex

    gw_idx = GatewayIndex.from_csv(gateway_csv_path)
    result = gw_idx.assign_gateway(x=129.0, y=144.0, floor_name="5F")
    df = gw_idx.assign_batch(df)  # 벡터화 배치 처리
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from scipy.spatial import KDTree

# ─── 상수 정의 ─────────────────────────────────────────────────────────
# 외부 시설 gateway_name 키워드 -> spot_name 매핑
# cp949 CSV의 깨진 한글을 UTF-8 변환 후 매핑
OUTDOOR_GATEWAY_KEYWORDS: dict[str, str] = {
    "타각기": "타각기 주변",
    "이스텔리커페테리": "공사현장",  # 이스텔리커 = 외부 식당 근처
    "SK전진사무소": "공사현장",
    "SK하이닉스사무동": "공사현장",
    "휴게실": "FAB 건너 휴게실",
    "흡연장": "FAB 건너 흡연실",  # 흡연장 = 흡연실
    "흡연실": "FAB 건너 흡연실",
    "화장실": "FAB 건너 화장실",  # 남녀화장실 공통
    "호이스트 2": "호이스트2 주변",
    "호이스트 4": "호이스트3 주변",  # 호이스트4 -> 호이스트3 매핑
    "양중화물용_컨테이너": "공사현장",
    "클라이머": "공사현장",  # 클라이머 엘리베이터
    "보안스티커": "공사현장",  # 보안스티커부착장소
}

# FAB 층별 Locus 매핑
FAB_FLOOR_LOCUS: dict[str, str] = {
    "5F": "L-M15X-001",
    "6F": "L-M15X-002",
    "7F": "L-M15X-003",
    "RF": "L-M15X-004",
}

# 외부 시설 spot_name -> locus_id 매핑
OUTDOOR_SPOT_LOCUS: dict[str, str] = {
    "타각기 주변": "L-M15X-009",
    "공사현장": "L-M15X-008",
    "FAB 건너 휴게실": "L-M15X-006",
    "FAB 건너 흡연실": "L-M15X-007",
    "FAB 건너 화장실": "L-M15X-005",
    "호이스트2 주변": "L-M15X-010",
    "호이스트3 주변": "L-M15X-011",
}


class GatewayIndex:
    """52개 S-Ward 좌표 기반 KDTree 인덱스."""

    def __init__(self, gw_df: pd.DataFrame):
        """
        Args:
            gw_df: GatewayPoint CSV DataFrame (UTF-8 인코딩)
                컬럼: gateway_no, gateway_name, building_no, building_name,
                      floor_no, floor_name, location_x, location_y
        """
        self._gw_df = gw_df.copy()
        self._gw_meta: dict[int, dict] = {}
        self._floor_trees: dict[str, "KDTree"] = {}
        self._floor_indices: dict[str, list[int]] = {}
        self._floor_coords: dict[str, np.ndarray] = {}
        self._outdoor_tree: "KDTree | None" = None
        self._outdoor_indices: list[int] = []
        self._outdoor_coords: np.ndarray | None = None

        self._build_index()

    def __len__(self) -> int:
        """전체 Gateway 수."""
        return len(self._gw_df)

    @property
    def floor_names(self) -> list[str]:
        """FAB 층 목록."""
        return list(self._floor_trees.keys())

    @property
    def outdoor_count(self) -> int:
        """외부 시설 Gateway 수."""
        return len(self._outdoor_indices)

    def _build_index(self) -> None:
        """층별 + 외부 KDTree 구축."""
        from scipy.spatial import KDTree

        # 메타데이터 구축
        for _, row in self._gw_df.iterrows():
            gw_no = int(row["gateway_no"])
            floor_name = row.get("floor_name")
            # NaN 처리
            floor_str = "" if pd.isna(floor_name) else str(floor_name).strip()

            self._gw_meta[gw_no] = {
                "gateway_name": str(row["gateway_name"]).strip(),
                "building_name": str(row.get("building_name", "")).strip()
                if pd.notna(row.get("building_name"))
                else "",
                "floor_name": floor_str,
                "x": float(row["location_x"]),
                "y": float(row["location_y"]),
            }

        # FAB 층별 인덱스
        fab_mask = self._gw_df["building_name"] == "FAB"
        fab_df = self._gw_df[fab_mask]

        for floor_name, group in fab_df.groupby("floor_name"):
            floor_str = str(floor_name).strip()
            coords = group[["location_x", "location_y"]].values.astype(np.float64)
            self._floor_trees[floor_str] = KDTree(coords)
            self._floor_indices[floor_str] = group["gateway_no"].astype(int).tolist()
            self._floor_coords[floor_str] = coords

        # 외부 시설 (building_name 없거나 FAB 아닌 것)
        outdoor_mask = (
            self._gw_df["building_name"].isna()
            | (self._gw_df["building_name"] == "")
            | (self._gw_df["building_name"] != "FAB")
        )
        outdoor_df = self._gw_df[outdoor_mask & ~fab_mask]

        if not outdoor_df.empty:
            coords = outdoor_df[["location_x", "location_y"]].values.astype(np.float64)
            self._outdoor_tree = KDTree(coords)
            self._outdoor_indices = outdoor_df["gateway_no"].astype(int).tolist()
            self._outdoor_coords = coords

    @classmethod
    def from_csv(cls, csv_path: Path, encoding: str = "cp949") -> "GatewayIndex":
        """
        CSV 파일에서 GatewayIndex 생성.

        Args:
            csv_path: GatewayPoint CSV 경로
            encoding: 파일 인코딩 (기본 cp949)

        Returns:
            GatewayIndex 인스턴스
        """
        if not csv_path.exists():
            raise FileNotFoundError(f"GatewayPoint CSV 없음: {csv_path}")

        df = pd.read_csv(csv_path, encoding=encoding)
        return cls(df)

    def assign_gateway(
        self,
        x: float,
        y: float,
        floor_name: str,
        building_name: str = "FAB",
    ) -> dict:
        """
        (x, y, floor) -> 가장 가까운 S-Ward 할당.

        Args:
            x: X 좌표
            y: Y 좌표
            floor_name: 층 이름 (5F, 6F, 7F, RF 또는 빈 문자열)
            building_name: 건물 이름 (기본 FAB)

        Returns:
            {
                "gateway_no": int,
                "gateway_name": str,
                "distance": float,
                "floor_name": str,
                "locus_id": str | None,
            }
        """
        floor_str = str(floor_name).strip() if pd.notna(floor_name) else ""

        # FAB 내부
        if building_name == "FAB" and floor_str in self._floor_trees:
            tree = self._floor_trees[floor_str]
            indices = self._floor_indices[floor_str]

            dist, idx = tree.query([x, y])
            gw_no = indices[idx]
            meta = self._gw_meta.get(gw_no, {})

            # FAB 층별 Locus 매핑
            locus_id = FAB_FLOOR_LOCUS.get(floor_str)

            return {
                "gateway_no": gw_no,
                "gateway_name": meta.get("gateway_name", f"GW-{gw_no}"),
                "distance": float(dist),
                "floor_name": floor_str,
                "locus_id": locus_id,
            }

        # 외부 시설
        if self._outdoor_tree is not None:
            dist, idx = self._outdoor_tree.query([x, y])
            gw_no = self._outdoor_indices[idx]
            meta = self._gw_meta.get(gw_no, {})

            # 외부 시설 Locus 매핑
            locus_id = self._resolve_outdoor_locus(meta.get("gateway_name", ""))

            return {
                "gateway_no": gw_no,
                "gateway_name": meta.get("gateway_name", f"GW-{gw_no}"),
                "distance": float(dist),
                "floor_name": "",
                "locus_id": locus_id,
            }

        # 매핑 실패
        return {
            "gateway_no": -1,
            "gateway_name": "unmapped",
            "distance": float("inf"),
            "floor_name": floor_str,
            "locus_id": None,
        }

    def _resolve_outdoor_locus(self, gateway_name: str) -> str | None:
        """외부 시설 gateway_name -> locus_id 매핑."""
        name_lower = gateway_name.lower()

        for keyword, spot_name in OUTDOOR_GATEWAY_KEYWORDS.items():
            if keyword.lower() in name_lower or keyword in gateway_name:
                return OUTDOOR_SPOT_LOCUS.get(spot_name)

        # 기본: 공사현장
        return OUTDOOR_SPOT_LOCUS.get("공사현장")

    def assign_batch(
        self,
        df: pd.DataFrame,
        x_col: str = "x",
        y_col: str = "y",
        level_col: str = "Level",
        building_col: str = "Building",
    ) -> pd.DataFrame:
        """
        DataFrame의 (x, y, Level) -> gateway_no, gateway_name 컬럼 추가.
        벡터화/KDTree 사용으로 고속 처리.

        Args:
            df: 입력 DataFrame
            x_col: X 좌표 컬럼명
            y_col: Y 좌표 컬럼명
            level_col: 층 컬럼명
            building_col: 건물 컬럼명

        Returns:
            추가 컬럼이 포함된 DataFrame:
                - gw_gateway_no: S-Ward 번호
                - gw_gateway_name: S-Ward 이름
                - gw_distance: 할당 거리
                - gw_locus_id: Locus ID
        """
        if df.empty:
            df = df.copy()
            df["gw_gateway_no"] = pd.Series(dtype=int)
            df["gw_gateway_name"] = pd.Series(dtype=str)
            df["gw_distance"] = pd.Series(dtype=float)
            df["gw_locus_id"] = pd.Series(dtype=str)
            return df

        df = df.copy()

        # 결과 배열 초기화
        n = len(df)
        gw_nos = np.full(n, -1, dtype=np.int32)
        gw_names = np.empty(n, dtype=object)
        distances = np.full(n, np.inf, dtype=np.float64)
        locus_ids = np.empty(n, dtype=object)

        # 좌표 추출
        x_vals = df[x_col].values.astype(np.float64)
        y_vals = df[y_col].values.astype(np.float64)

        # Building, Level 컬럼 처리 (없으면 기본값)
        if building_col in df.columns:
            buildings = df[building_col].fillna("").astype(str).values
        else:
            buildings = np.array(["FAB"] * n)

        if level_col in df.columns:
            levels = df[level_col].fillna("").astype(str).values
        else:
            levels = np.array([""] * n)

        # FAB 층별 벡터화 처리
        for floor_str, tree in self._floor_trees.items():
            # 해당 층 마스크
            mask = (buildings == "FAB") & (levels == floor_str)
            if not mask.any():
                continue

            indices_floor = self._floor_indices[floor_str]
            coords = np.column_stack([x_vals[mask], y_vals[mask]])

            # KDTree 배치 쿼리
            dists, idxs = tree.query(coords)

            # 결과 할당
            gw_nos[mask] = [indices_floor[i] for i in idxs]
            gw_names[mask] = [
                self._gw_meta.get(indices_floor[i], {}).get("gateway_name", f"GW-{indices_floor[i]}")
                for i in idxs
            ]
            distances[mask] = dists
            locus_ids[mask] = FAB_FLOOR_LOCUS.get(floor_str, "")

        # 외부 시설 처리
        if self._outdoor_tree is not None:
            # 아직 할당 안된 레코드 (외부 또는 FAB 아닌 건물)
            outdoor_mask = (gw_nos == -1) | (buildings != "FAB")
            outdoor_mask &= (levels == "") | pd.isna(df[level_col]) if level_col in df.columns else True

            if outdoor_mask.any():
                coords = np.column_stack([x_vals[outdoor_mask], y_vals[outdoor_mask]])
                dists, idxs = self._outdoor_tree.query(coords)

                # 결과 할당
                gw_nos[outdoor_mask] = [self._outdoor_indices[i] for i in idxs]
                gw_names[outdoor_mask] = [
                    self._gw_meta.get(self._outdoor_indices[i], {}).get(
                        "gateway_name", f"GW-{self._outdoor_indices[i]}"
                    )
                    for i in idxs
                ]
                distances[outdoor_mask] = dists

                # 외부 시설 Locus 매핑
                for i, idx in enumerate(np.where(outdoor_mask)[0]):
                    gw_name = gw_names[idx]
                    locus_ids[idx] = self._resolve_outdoor_locus(str(gw_name))

        df["gw_gateway_no"] = gw_nos
        df["gw_gateway_name"] = gw_names
        df["gw_distance"] = distances
        df["gw_locus_id"] = locus_ids

        return df

    def get_gateway_info(self, gateway_no: int) -> dict:
        """S-Ward 메타데이터 조회."""
        return self._gw_meta.get(gateway_no, {})

    def get_floor_gateways(self, floor_name: str) -> list[dict]:
        """특정 층의 S-Ward 목록."""
        floor_str = str(floor_name).strip()
        if floor_str not in self._floor_indices:
            return []

        result = []
        for gw_no in self._floor_indices[floor_str]:
            info = self._gw_meta.get(gw_no, {})
            result.append({
                "gateway_no": gw_no,
                "gateway_name": info.get("gateway_name", ""),
                "x": info.get("x", 0),
                "y": info.get("y", 0),
            })
        return result

    def get_spot_for_gateway(self, gateway_no: int) -> str:
        """S-Ward -> 소속 Spot 이름 반환 (11개 Spot 기준)."""
        meta = self._gw_meta.get(gateway_no)
        if meta is None:
            return "Unknown"

        floor_str = meta.get("floor_name", "")
        building = meta.get("building_name", "")

        # FAB 내부
        if building == "FAB" and floor_str in FAB_FLOOR_LOCUS:
            return f"FAB {floor_str} (전체)"

        # 외부 시설
        gw_name = meta.get("gateway_name", "")
        for keyword, spot_name in OUTDOOR_GATEWAY_KEYWORDS.items():
            if keyword in gw_name:
                return spot_name

        return "공사현장"

    def get_outdoor_gateways(self) -> list[dict]:
        """외부 시설 S-Ward 목록."""
        result = []
        for gw_no in self._outdoor_indices:
            info = self._gw_meta.get(gw_no, {})
            result.append({
                "gateway_no": gw_no,
                "gateway_name": info.get("gateway_name", ""),
                "x": info.get("x", 0),
                "y": info.get("y", 0),
                "spot_name": self.get_spot_for_gateway(gw_no),
            })
        return result

    def to_dataframe(self) -> pd.DataFrame:
        """전체 Gateway 정보를 DataFrame으로 반환."""
        records = []
        for gw_no, meta in self._gw_meta.items():
            records.append({
                "gateway_no": gw_no,
                "gateway_name": meta.get("gateway_name", ""),
                "building_name": meta.get("building_name", ""),
                "floor_name": meta.get("floor_name", ""),
                "x": meta.get("x", 0),
                "y": meta.get("y", 0),
                "spot_name": self.get_spot_for_gateway(gw_no),
            })
        return pd.DataFrame(records)


# ─── 헬퍼 함수 ─────────────────────────────────────────────────────────
def load_gateway_index(sector_id: str | None = None) -> GatewayIndex:
    """
    config에서 GatewayPoint CSV 경로 조회 후 GatewayIndex 반환.

    Args:
        sector_id: Sector ID (기본값 사용 시 None)

    Returns:
        GatewayIndex 인스턴스
    """
    import config as cfg

    paths = cfg.get_sector_paths(sector_id or cfg.SECTOR_ID)
    gw_path = paths.get("gateway_raw_csv")

    if not gw_path or not gw_path.exists():
        raise FileNotFoundError(f"GatewayPoint CSV 없음: {gw_path}")

    return GatewayIndex.from_csv(gw_path, encoding=cfg.RAW_ENCODING)


def save_gateway_csv_utf8(
    src_path: Path,
    dst_path: Path,
    src_encoding: str = "cp949",
) -> None:
    """
    cp949 원본 CSV를 UTF-8로 변환하여 저장.

    Args:
        src_path: 원본 CSV 경로 (cp949)
        dst_path: 저장 경로 (UTF-8)
        src_encoding: 원본 인코딩
    """
    df = pd.read_csv(src_path, encoding=src_encoding)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False, encoding="utf-8-sig")
