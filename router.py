import json
from utils import run_rag_pipline, graph_rag_pipline
from model import RagLLM
from prompt_cfg import TOOL_DESC, REACT_PROMPT

class Tools:
    def __init__(self) -> None:
        self.toolConfig = self._tools()
    
    def _tools(self):
        tools = [
            {
                'name_for_human': '查询公司规章制度的工具',
                'name_for_model': 'get_guizha',
                'description_for_model': '获取公司的相关规章制度，包括考勤、工作时间、请假、出差费用规定',
                'parameters': []
            },
            {
                'name_for_human': '查询企业、金融和商业的工具',
                'name_for_model': 'get_finance',
                'description_for_model': '获取企业相关的信息包括经营事项和危机事件以及企业的投资者信息等',
                'parameters': []
            },
            {
                'name_for_human': '查询其他问题的工具',
                'name_for_model': 'other',
                'description_for_model': '获取其他问题的信息等',
                'parameters': []
            }
        ]
        return tools

    def get_guizha(self, query):
        return run_rag_pipline(query, query, stream=True)
    
    def get_finance(self, query):
        return graph_rag_pipline(query, stream=True)
    def other(self, query):
        return "对不起，我不能回答这个问题"

class Agent:
    def __init__(self) -> None:
        self.tool = Tools()
        self.model = RagLLM()
        self.system_prompt = self.build_system_input()
        
    def build_system_input(self):
        tool_descs, tool_names = [], []
        for tool in self.tool.toolConfig:
            tool_descs.append(TOOL_DESC.format(**tool))
            tool_names.append(tool['name_for_model'])
        tool_descs = '\n\n'.join(tool_descs)
        tool_names = ','.join(tool_names)
        sys_prompt = REACT_PROMPT.format(tool_descs=tool_descs, tool_names=tool_names)
        return sys_prompt
    
    def parse_latest_plugin_call(self, text):
        plugin_name, plugin_args = '', ''
        i = text.rfind('\nAction:')
        j = text.rfind('\nAction Input:')
        k = text.rfind('\nObservation:')
        if 0 <= i < j:  # If the text has `Action` and `Action input`,
            if k < j:  # but does not contain `Observation`,
                text = text.rstrip() + '\nObservation:'  # Add it back.
            k = text.rfind('\nObservation:')
            plugin_name = text[i + len('\nAction:') : j].strip()
            plugin_args = text[j + len('\nAction Input:') : k].strip()
            text = text[:k]
        return plugin_name, plugin_args, text
    
    def call_plugin(self, plugin_name, plugin_args, ori_text):
        
        try:
            plugin_args = json.loads(plugin_args)
        except:
            pass
        if plugin_name == 'get_guizha':
            return self.tool.get_guizha(ori_text)
        if plugin_name == 'get_finance':
            return self.tool.get_finance(ori_text)
        if plugin_name == 'other':
            return self.tool.other(ori_text)
        
    def text_completion(self, text, history=[]):
        ori_text = text
        text = "\nQuestion:" + text
        response = self.model(f"{self.system_prompt} \n {text}")
        plugin_name, plugin_args, response = self.parse_latest_plugin_call(response)

        if plugin_name:
            return self.call_plugin(plugin_name, plugin_args, ori_text)
        return "对不起，我不能回答这个问题"