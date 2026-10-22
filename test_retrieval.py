import unittest
from retrieval import ingest, search, build_graph

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        self.chunks = ingest([('deploy.md','发布失败后回滚至稳定版本。'),('cache.md','缓存穿透可以用空值缓存缓解。')])
    def test_rank(self):
        self.assertEqual(search('发布回滚',self.chunks)[0]['source'],'deploy.md')
    def test_dedup(self):
        self.assertEqual(len(ingest([('a','same'),('b','same')])),1)
    def test_chunk(self):
        result = ingest([('a','abcdefghij')],6,2)
        self.assertEqual(result[1]['offset'],4)
        self.assertEqual(result[1]['text'],'efghij')
    def test_reject(self):
        result = build_graph(self.chunks).invoke({'question':'火星天气'})
        self.assertEqual(result['trace'],['retrieve','reject'])
    def test_citation(self):
        result = build_graph(self.chunks).invoke({'question':'缓存'})
        self.assertIn('[1] cache.md',result['answer'])
    def test_model_path(self):
        class Stub:
            def invoke(self,messages):
                assert '[1]' in messages[1][1]
                return type('Response',(),{'content':'依据资料 [1]'})()
        self.assertEqual(build_graph(self.chunks,Stub()).invoke({'question':'缓存'})['answer'],'依据资料 [1]')
    def test_empty(self):
        self.assertEqual(search('',self.chunks),[])
    def test_bad_overlap(self):
        with self.assertRaises(ValueError): ingest([],5,5)

if __name__ == '__main__': unittest.main()
