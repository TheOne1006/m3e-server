import unittest
import numpy as np
from m3e_server import text_embeddings, enableModels, cacheModels
from sentence_transformers import SentenceTransformer


sentences = [
    '* Moka 此文本嵌入模型由 MokaAI 训练并开源，训练脚本使用 uniem',
    '* Massive 此文本嵌入模型通过**千万级**的中文句对数据集进行训练',
    '* Mixed 此文本嵌入模型支持中英双语的同质文本相似度计算，异质文本检索等功能，未来还会支持代码检索，ALL in one'
]


def get_np_expected(model, sentences: list[str]):
    expected = model.encode(sentences)
    np_expected = np.array([embedding / np.linalg.norm(embedding) for embedding in expected])
    
    return np_expected



class MyTestCase(unittest.TestCase):
    
    @classmethod
    def setUp(cls) -> None:
        cacheModels.clear()
        enableModels.clear()
        
    def test_small_embedding(self):
        embadding = "m3e-small"
        model = SentenceTransformer(f"moka-ai/{embadding}")
        enableModels.append(f"moka-ai/{embadding}")
        
        actual = text_embeddings(embadding, sentences)
        expected = get_np_expected(model, sentences)
        
        actual = [item['embedding'] for item in actual['data']]
        np_actual = np.array(actual)
        assert np.allclose(np_actual, expected), 'small embedding not equal'
    
    def test_small_embedding_chang_dim(self):
        embadding = "m3e-small"
        
        _ = SentenceTransformer(f"moka-ai/{embadding}")
        enableModels.append(f"moka-ai/{embadding}")
        
        actual = text_embeddings(embadding, sentences, 1024)
        
        actual = [item['embedding'] for item in actual['data']]
        np_actual = np.array(actual)
        assert np_actual.shape == (3, 1024)


if __name__ == '__main__':
    unittest.main()
