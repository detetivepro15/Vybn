import numpy as np
import cmath
from sentence_transformers import SentenceTransformer

_E = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def estado(texto: str) -> np.ndarray:
    v = _E.encode(texto)
    v = v / np.linalg.norm(v)
    z = v[0::2] + 1j * v[1::2]
    return z / np.sqrt(np.sum(np.abs(z)**2))

def avaliar(m: np.ndarray, x: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    theta = cmath.phase(np.vdot(m, x))
    m2 = alpha * m + (1 - alpha) * x * cmath.exp(1j * theta)
    n = np.sqrt(np.sum(np.abs(m2)**2))
    return m2 / n if n > 1e-10 else m2

def fidelidade(a: np.ndarray, b: np.ndarray) -> float:
    return float(abs(np.vdot(a, b))**2)

def centroid(textos: list[str]) -> np.ndarray:
    xs = [estado(t) for t in textos]
    K = np.mean(xs, axis=0)
    return K / np.sqrt(np.sum(np.abs(K)**2))

def lente(M: np.ndarray, texto: str, alpha: float = 0.5, eps: float = 0.1) -> dict:
    neutro = (
        "A previsão do tempo indica temperaturas amenas.",
        "O formulário padrão foi preenchido dentro do prazo.",
        "A loja repõe os estoques às terças-feiras."
    )
    x = estado(texto)
    Mx = avaliar(M, x, alpha)
    lacunas = [1 - fidelidade(Mx, avaliar(M, estado(n), alpha)) for n in neutro]
    
    return {
        "theta": cmath.phase(np.vdot(M, x)),
        "acoplamento": float(abs(np.vdot(M, x))),
        "distintividade": 1 - fidelidade(M, x),
        "rotação": 1 - fidelidade(M, Mx),
        "taxa_de_rotação": (1 - fidelidade(M, avaliar(M, x, eps))) / eps,
        "counterfactual_gap": float(np.median(lacunas))
    }

def mesclagem(A: np.ndarray, B: np.ndarray, alpha: float = 0.5, iters: int = 50, tol: float = 1e-8) -> np.ndarray:
    a, b, prev = A.copy(), B.copy(), None
    for _ in range(iters):
        a, b = avaliar(a, b, alpha), avaliar(b, a, alpha)
        fp = (a + b) / 2
        fp /= np.sqrt(np.sum(np.abs(fp)**2))
        if prev is not None and np.linalg.norm(fp - prev) < tol:
            break
        prev = fp
    return fp
