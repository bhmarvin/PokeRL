import json
import math
from typing import Dict, Any, List, Optional

from poke_env.data import to_id_str

# Constants for stat calculation
# Final Stat = floor(floor((2 * Base + IV + floor(EV/4)) * Level / 100 + 5) * Nature)
# HP = floor((2 * Base + IV + floor(EV/4)) * Level / 100 + Level + 10)

class RandbatsMeta:
    def __init__(self, data_path: str = "data/gen9randombattle_stats.json"):
        with open(data_path, "r") as f:
            raw_data = json.load(f)
        self.data = {
            self._normalize_name(species): self._normalize_species_data(spec_data)
            for species, spec_data in raw_data.items()
        }

    def _normalize_name(self, value: Optional[str]) -> str:
        return to_id_str(value or "")

    def _normalize_distribution(self, value: Dict[str, Any]) -> Dict[str, Any]:
        return {
            self._normalize_name(str(key)): item
            for key, item in value.items()
        }

    def _normalize_role_data(self, role_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(role_data)
        if "moves" in normalized:
            normalized["moves"] = self._normalize_distribution(normalized["moves"])
        if "items" in normalized:
            normalized["items"] = self._normalize_distribution(normalized["items"])
        if "abilities" in normalized:
            normalized["abilities"] = self._normalize_distribution(normalized["abilities"])
        if "teraTypes" in normalized:
            normalized["teraTypes"] = self._normalize_distribution(normalized["teraTypes"])
        return normalized

    def _normalize_species_data(self, spec_data: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(spec_data)
        if "items" in normalized:
            normalized["items"] = self._normalize_distribution(normalized["items"])
        if "abilities" in normalized:
            normalized["abilities"] = self._normalize_distribution(normalized["abilities"])
        if "roles" in normalized:
            normalized["roles"] = {
                role_name: self._normalize_role_data(role_data)
                for role_name, role_data in normalized["roles"].items()
            }
        return normalized
            
    def get_species_data(self, species: str) -> Optional[Dict[str, Any]]:
        return self.data.get(self._normalize_name(species))

    def calculate_stat(self, base: int, level: int, ev: int = 0, iv: int = 31, nature: float = 1.0, is_hp: bool = False) -> int:
        if is_hp:
            # Shedinja is 1 HP always
            if base == 1: return 1
            return math.floor((2 * base + iv + math.floor(ev / 4)) * level / 100 + level + 10)
        else:
            return math.floor(math.floor((2 * base + iv + math.floor(ev / 4)) * level / 100 + 5) * nature)

    def get_role_stats(self, species_name: str, role_name: str, base_stats: Dict[str, int]) -> Dict[str, int]:
        spec = self.get_species_data(species_name)
        if not spec: return {}
        
        level = spec.get("level", 80)
        role = spec["roles"][role_name]
        evs = role.get("evs", {})
        ivs = role.get("ivs", {})
        
        # Default IVs in Randbats are 31 for everything (usually)
        # Default EVs are 84 in everything unless specified
        calculated = {}
        for stat in ["hp", "atk", "def", "spa", "spd", "spe"]:
            base = base_stats.get(stat, 100)
            ev = evs.get(stat, 84) # Standard spread if not defined
            iv = ivs.get(stat, 31)
            calculated[stat] = self.calculate_stat(base, level, ev, iv, is_hp=(stat == "hp"))
            
        return calculated

    def filter_roles(self, species_name: str, revealed_moves: List[str], revealed_item: Optional[str] = None) -> Dict[str, float]:
        spec = self.get_species_data(species_name)
        if not spec: return {}
        normalized_moves = [self._normalize_name(move) for move in revealed_moves]
        normalized_item = self._normalize_name(revealed_item) if revealed_item else None
        if normalized_item == "unknownitem":
            normalized_item = None

        valid_roles = {}
        total_weight = 0.0
        
        for role_name, role_data in spec["roles"].items():
            possible = True
            
            # Check moves
            role_moves = role_data.get("moves", {})
            for move in normalized_moves:
                if move not in role_moves:
                    possible = False
                    break
            
            # Check items
            if possible and normalized_item:
                role_items = role_data.get("items", {})
                if normalized_item not in role_items:
                    possible = False
            
            if possible:
                weight = role_data.get("weight", 1.0)
                valid_roles[role_name] = weight
                total_weight += weight
                
        # Normalize weights to 1.0
        if total_weight > 0:
            for role in valid_roles:
                valid_roles[role] /= total_weight
                
        return valid_roles

    def get_role_move_distribution(self, species_name: str, role_name: str) -> Dict[str, float]:
        spec = self.get_species_data(species_name)
        if not spec:
            return {}
        role = spec.get("roles", {}).get(role_name)
        if not role:
            return {}
        return {
            move_id: float(move_prob)
            for move_id, move_prob in role.get("moves", {}).items()
        }

    def get_role_item_distribution(self, species_name: str, role_name: str) -> Dict[str, float]:
        spec = self.get_species_data(species_name)
        if not spec:
            return {}
        role = spec.get("roles", {}).get(role_name)
        if not role:
            return {}
        return {
            item_id: float(item_prob)
            for item_id, item_prob in role.get("items", {}).items()
        }

    def get_move_marginals(self, species_name: str, role_weights: Dict[str, float]) -> Dict[str, float]:
        marginals: Dict[str, float] = {}
        for role_name, role_weight in role_weights.items():
            for move_id, move_prob in self.get_role_move_distribution(species_name, role_name).items():
                marginals[move_id] = marginals.get(move_id, 0.0) + float(role_weight) * float(move_prob)
        return {
            move_id: min(1.0, max(0.0, move_prob))
            for move_id, move_prob in marginals.items()
        }

    def get_item_marginals(self, species_name: str, role_weights: Dict[str, float]) -> Dict[str, float]:
        marginals: Dict[str, float] = {}
        for role_name, role_weight in role_weights.items():
            for item_id, item_prob in self.get_role_item_distribution(species_name, role_name).items():
                marginals[item_id] = marginals.get(item_id, 0.0) + float(role_weight) * float(item_prob)
        return {
            item_id: min(1.0, max(0.0, item_prob))
            for item_id, item_prob in marginals.items()
        }
