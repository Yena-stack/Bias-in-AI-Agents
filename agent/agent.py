import memory
import llm


class ConversationalAgent:
    def __init__(self, agent_id: int, mem_window: int = 3, sys_prompt: llm.Message = None, temp: float = 1.0):
        self.agent_id = agent_id
        self.mem_window = mem_window
        self.sys_prompt = sys_prompt
        self.mem = memory.Memory()
        self.temp = temp
        
    def respond(self, msg: llm.Message, **kwargs) -> llm.Message:
        hist = self.mem.retrieve(time=msg.time - self.mem_window)        
        msgs = [self.sys_prompt] + [llm.Message(time=m.time, content=m.content, role=m.role) for m in hist] + [msg]
        reply = llm.chat_request(messages=msgs, temperature=self.temp, **kwargs)
        self.mem.store(message=msg)
        self.mem.store(message=reply)
        return reply


class ProbabilityAgent:
    """Agent for outputting token distributions with logprobs"""
    def __init__(self, agent_id: int, mem_window: int = 0, sys_prompt: llm.Message = None, temp: float = 1.0):
        self.agent_id = agent_id
        self.mem_window = mem_window
        self.sys_prompt = sys_prompt
        self.mem = memory.Memory()
        self.temp = temp

    def generate(self, msg: llm.Message, token_limit: int = 10, top_probs: int = 5, **kwargs) -> tuple[dict, any]:
        hist = self.mem.retrieve(time=msg.time - self.mem_window)
        msgs = [self.sys_prompt] + hist + [msg]
        reply, probs = llm.complete_request(messages=msgs, max_tokens=token_limit, temperature=self.temp, logprobs=top_probs, **kwargs)
        self.mem.store(message=msg)
        self.mem.store(message=reply)
        return reply, probs