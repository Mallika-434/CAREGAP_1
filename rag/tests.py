from unittest.mock import patch

from django.test import SimpleTestCase
from requests.exceptions import ConnectionError

from rag.pipeline import RAGPipeline


class RAGPipelineFallbackTests(SimpleTestCase):
    def setUp(self):
        self.pipeline = RAGPipeline()

    @patch.dict(
        'os.environ',
        {
            'MEDGEMMA_URL': 'http://localhost:8001/v1/chat/completions',
            'MEDGEMMA_MODEL': 'google/medgemma-1.5-4b-it',
        },
        clear=False,
    )
    @patch.object(RAGPipeline, 'retrieve', return_value=[])
    @patch.object(RAGPipeline, '_call_medgemma', return_value='Medical suggestion')
    @patch.object(RAGPipeline, '_call_ollama', return_value='Local suggestion')
    @patch.object(RAGPipeline, '_call_gemini')
    def test_generate_suggestions_prefers_medgemma(
        self,
        gemini_mock,
        ollama_mock,
        medgemma_mock,
        _retrieve_mock,
    ):
        result = self.pipeline.generate_suggestions({'name': 'Alice', 'age': 52})

        self.assertEqual(result['suggestions'], 'Medical suggestion')
        self.assertEqual(result['model'], 'medgemma-google/medgemma-1.5-4b-it')
        gemini_mock.assert_not_called()
        ollama_mock.assert_not_called()
        medgemma_mock.assert_called_once()

    @patch.dict('os.environ', {'GEMINI_API_KEY': 'test-key'}, clear=False)
    @patch.object(RAGPipeline, 'retrieve', return_value=[])
    @patch.object(RAGPipeline, '_call_medgemma', side_effect=ConnectionError('offline'))
    @patch.object(RAGPipeline, '_call_ollama', side_effect=ConnectionError('offline'))
    @patch.object(RAGPipeline, '_call_gemini', return_value='Gemini suggestion')
    def test_generate_suggestions_uses_gemini_when_local_models_fail(
        self,
        gemini_mock,
        _medgemma_mock,
        _ollama_mock,
        _retrieve_mock,
    ):
        result = self.pipeline.generate_suggestions({'name': 'Alice', 'age': 52})

        self.assertEqual(result['suggestions'], 'Gemini suggestion')
        self.assertEqual(result['model'], 'gemini-fallback')
        gemini_mock.assert_called_once()

    @patch.object(RAGPipeline, '_call_medgemma', side_effect=ConnectionError('offline'))
    @patch.object(RAGPipeline, '_call_ollama', side_effect=ConnectionError('offline'))
    @patch.object(RAGPipeline, '_call_gemini', return_value=None)
    def test_explain_patient_result_falls_back_to_rule_based(
        self,
        gemini_mock,
        _medgemma_mock,
        _ollama_mock,
    ):
        result = self.pipeline.explain_patient_result(
            'bmi_assessment',
            {
                'name': 'Sam',
                'age': 12,
                'gender': 'M',
                'bmi': 28.4,
                'category': 'Overweight',
                'recommendation': 'Schedule nutrition counseling.',
            },
        )

        self.assertEqual(result['source'], 'rule_based')
        self.assertIn('BMI of 28.4', result['explanation'])
        gemini_mock.assert_called_once()

    @patch.object(RAGPipeline, 'retrieve', return_value=[])
    @patch.object(RAGPipeline, '_call_gemini', return_value=None)
    def test_demo_mode_skips_local_llms(
        self,
        gemini_mock,
        _retrieve_mock,
    ):
        with self.settings(DEPLOYMENT_MODE='demo'):
            result = self.pipeline.generate_suggestions({'name': 'Alice', 'age': 52})

        self.assertEqual(result['model'], 'rule-based-fallback')
        gemini_mock.assert_called_once()
