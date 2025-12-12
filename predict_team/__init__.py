import json
import sys
from pathlib import Path
import numpy as np
import onnxruntime as ort
import azure.functions as func

# Ensure project root is importable (for src.* modules)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from shared.pokedex import ALL_POKEMON, NAME_TO_IDX, NUM_POKEMON


class OnnxLivePredictor:
    def __init__(self, onnx_path: Path | None = None):
        self.all_pokemon = ALL_POKEMON
        self.name_to_idx = NAME_TO_IDX

        # Input dim must match training: 2 (meta) + 4 * NUM_POKEMON (one-hot vectors)
        self.input_dim = 2 + (NUM_POKEMON * 4)

        model_path = onnx_path or (PROJECT_ROOT / "models" / "pokemon_predictor.onnx")
        if not model_path.exists():
            raise FileNotFoundError(f"ONNX model not found at {model_path}")

        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])  # CPU by default
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def _encode_pokemon(self, pokemon_name):
        """Converts a single pokemon name to a one-hot vector."""
        one_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
        idx = self.name_to_idx.get(pokemon_name, -1)
        if idx != -1:
            one_hot[idx] = 1.0
        return one_hot

    def _encode_team(self, team_list):
        """Converts a list of pokemon (strings or dicts) to a multi-hot vector."""
        multi_hot = np.zeros(NUM_POKEMON, dtype=np.float32)
        for item in team_list:
            # Handle standard string list or the dict structure from logs
            name = item['species'] if isinstance(item, dict) else item
            idx = self.name_to_idx.get(name, -1)
            if idx != -1:
                multi_hot[idx] = 1.0
        return multi_hot

    def predict(self, game_state: dict) -> list[tuple[str, float]]:
        # 1. Feature Engineering (same scaling as training)
        rating = float(game_state.get('rating', 1500)) / 2000.0
        turn = float(game_state.get('turn', 0)) / 50.0

        obs_active = self._encode_pokemon(game_state.get('my_active'))
        obs_team = self._encode_team(game_state.get('my_team', []))
        opp_active = self._encode_pokemon(game_state.get('opp_active'))
        opp_revealed = self._encode_team(game_state.get('opp_revealed', []))

        # 2. Build feature vector
        features = np.concatenate([
            np.array([rating, turn], dtype=np.float32),
            obs_active,
            obs_team,
            opp_active,
            opp_revealed,
        ]).astype(np.float32)

        features_batch = features.reshape(1, -1)

        # 3. Inference (ONNX outputs logits)
        logits = self.session.run(None, {self.input_name: features_batch})[0][0]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        # 4. Decode results, filter out already revealed opponent mons
        revealed_names = set()
        for item in game_state.get('opp_revealed', []):
            if isinstance(item, dict):
                nm = item.get('species')
                if nm:
                    revealed_names.add(nm)
            elif isinstance(item, str):
                revealed_names.add(item)

        results: list[tuple[str, float]] = []
        for idx, prob in enumerate(probs.tolist()):
            name = self.all_pokemon[idx]
            if name not in revealed_names:
                results.append((name, float(prob)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results


# Lazy-load singleton predictor to avoid reloading model per request
_PREDICTOR: OnnxLivePredictor | None = None

def get_predictor() -> OnnxLivePredictor:
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = OnnxLivePredictor()
    return _PREDICTOR


def _map_request_to_state(payload: dict) -> dict:
    """Map incoming JSON schema to the model's expected game_state."""
    meta = payload.get('meta', {})
    perspective = meta.get('perspective', 'p1')

    obs = payload.get('observer_state', {})
    opp = payload.get('opponent_state', {})

    # If the perspective is p2, swap roles for 'my' vs 'opp'
    my_state = obs if perspective == 'p1' else opp
    opp_state = opp if perspective == 'p1' else obs

    return {
        'rating': meta.get('rating', 1500),
        'turn': meta.get('turn_number', 0),
        'my_active': my_state.get('active_pokemon'),
        'my_team': my_state.get('revealed_team', []),
        'opp_active': opp_state.get('active_pokemon'),
        'opp_revealed': opp_state.get('revealed_team', []),
    }


def main(req: func.HttpRequest) -> func.HttpResponse:
    try:
        payload = req.get_json()
    except ValueError:
        return func.HttpResponse(
            json.dumps({"error": "Invalid JSON body"}),
            status_code=400,
            mimetype="application/json",
        )

    # Optional top-k override via query string or body
    k = req.params.get('k')
    if k is None:
        k = payload.get('k')
    try:
        k = int(k) if k is not None else 6
        if k <= 0:
            k = 6
    except Exception:
        k = 6

    try:
        predictor = get_predictor()
        game_state = _map_request_to_state(payload)
        ranked = predictor.predict(game_state)
        topk = ranked[:k]

        response = {
            "meta": {
                "game_id": payload.get("meta", {}).get("game_id"),
                "turn_number": payload.get("meta", {}).get("turn_number"),
                "perspective": payload.get("meta", {}).get("perspective"),
            },
            "k": k,
            "predictions": [
                {"species": name, "probability": prob} for name, prob in topk
            ],
        }
        return func.HttpResponse(
            json.dumps(response), status_code=200, mimetype="application/json"
        )
    except FileNotFoundError as e:
        return func.HttpResponse(
            json.dumps({"error": str(e)}), status_code=500, mimetype="application/json"
        )
    except Exception as e:
        # In production, log exception details; keep response minimal
        return func.HttpResponse(
            json.dumps({"error": "Prediction failed", "detail": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
