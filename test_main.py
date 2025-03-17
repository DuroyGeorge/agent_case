import unittest
from unittest.mock import patch, MagicMock, AsyncMock
import asyncio
import io
import sys
from main import main, async_main


class TestMain(unittest.TestCase):
    """测试main.py中的main函数和async_main函数"""

    @patch('asyncio.run')
    def test_main(self, mock_run):
        """测试main函数是否正确调用了asyncio.run(async_main())"""
        # 执行main函数
        main()
        
        # 验证asyncio.run是否被调用，并且参数是async_main()
        mock_run.assert_called_once()
        # 获取调用asyncio.run时的第一个参数
        call_args = mock_run.call_args[0][0]
        # 验证该参数是async_main协程对象
        self.assertTrue(asyncio.iscoroutine(call_args))


class TestAsyncMain(unittest.IsolatedAsyncioTestCase):
    """测试async_main函数的异步行为"""
    
    @patch('main.paper_main')
    @patch('main.pdf_process_main')
    @patch('main.summary')
    @patch('main.embedding')
    @patch('main.get_top_pairs')
    @patch('main.report_title')
    @patch('main.report_abstract')
    @patch('main.report_introduction')
    @patch('main.report_body')
    @patch('main.report_discussion')
    @patch('main.report_conclusion')
    @patch('builtins.open', new_callable=MagicMock)
    @patch('builtins.input')
    async def test_async_main_workflow(self, mock_input, mock_open, mock_report_conclusion, 
                                 mock_report_discussion, mock_report_body, mock_report_introduction, 
                                 mock_report_abstract, mock_report_title, mock_get_top_pairs, 
                                 mock_embedding, mock_summary, mock_pdf_process, mock_paper_main):
        """测试async_main函数的工作流程"""
        # 设置模拟输入
        mock_input.side_effect = ["人工智能", "5"]  # 第一次调用返回主题，第二次调用返回论文数量
        
        # 设置模拟返回值
        mock_papers = [{"title": "Paper1"}, {"title": "Paper2"}]
        mock_paper_main.return_value = mock_papers
        mock_pdf_process.return_value = mock_papers
        
        # 模拟get_top_pairs的返回值
        mock_get_top_pairs.return_value = [(0, 1)]
        
        # 模拟report系列函数的返回值
        mock_report_title.return_value = "# Title Section"
        mock_report_abstract.return_value = "## Abstract Section"
        mock_report_introduction.return_value = "## Introduction Section"
        mock_report_body.return_value = "## Body Section"
        mock_report_discussion.return_value = "## Discussion Section"
        mock_report_conclusion.return_value = "## Conclusion Section"
        
        # 模拟文件写入
        mock_file = MagicMock()
        mock_open.return_value.__enter__.return_value = mock_file
        
        # 执行async_main函数
        await async_main()
        
        # 验证调用
        mock_paper_main.assert_called_once_with("人工智能", 5)
        mock_pdf_process.assert_called_once_with(mock_papers)
        mock_summary.assert_called_once()
        mock_embedding.assert_called_once_with(mock_papers)
        
        # 验证get_top_pairs被调用了6次（对应6个部分）
        self.assertEqual(mock_get_top_pairs.call_count, 6)
        
        # 验证report函数都被调用了
        mock_report_title.assert_called_once()
        mock_report_abstract.assert_called_once()
        mock_report_introduction.assert_called_once()
        mock_report_body.assert_called_once()
        mock_report_discussion.assert_called_once()
        mock_report_conclusion.assert_called_once()
        
        # 验证文件写入
        mock_open.assert_called_once_with("report.md", "w")
        mock_file.write.assert_called_once()

    @patch('builtins.input')
    @patch('main.paper_main')
    @patch('main.pdf_process_main')
    async def test_async_main_with_invalid_paper_number(self, mock_pdf_process, mock_paper_main, mock_input):
        """测试输入无效的论文数量时的处理"""
        # 设置模拟输入：第一次调用返回主题，第二次调用返回无效的论文数量
        mock_input.side_effect = ["人工智能", "invalid"]
        
        # 设置模拟返回值
        mock_papers = [{"title": "Paper1"}]
        mock_paper_main.return_value = mock_papers
        mock_pdf_process.return_value = mock_papers
        
        # 屏蔽所有后续步骤的执行
        with patch('main.summary'), patch('main.embedding'), patch('main.get_top_pairs'), \
             patch('asyncio.gather'), patch('builtins.open'):
            # 执行async_main函数
            await async_main()
            
            # 验证paper_main被调用时使用了默认值10
            mock_paper_main.assert_called_once_with("人工智能", 10)


if __name__ == '__main__':
    unittest.main()
