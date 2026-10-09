"""Prospective public derivative; NOT used by the NAV7 learner.

This import relocation preserves the staged original argument-refusal
semantics. It does not claim executed training, paper validation or returns.
"""
"""Prospective visible tool-argument errors; never repairs or replays an action.

Only strict native frames with a declared recipient and matching JSON-object
command may reach this path. Argument schema refusals are exact local tool
results. Genuine framing/JSON/address failures remain fatal. The caller owns
sampling, accounting and actual SDK lifecycle; this file opens none.
"""
from copy import deepcopy
from training.native_agent.runtime import LocalGenerationBudgetExhausted,canonical,utc_now,write_new
from training.native_agent.compact_presentation import exact_json_equal
from training.framework import declared_tools as original

ERROR_STATUS='DECLARED_TOOL_ARGUMENT_VALIDATION_REFUSED'
ARGUMENT_ERROR_PREFIXES=('declared schema type mismatch: arguments','declared schema enum mismatch: arguments',
 'declared schema numeric bound mismatch: arguments','missing required declared fields: arguments','extra undeclared fields: arguments')

def argument_error(action,tools):
 """Return only the original known argument validator message; otherwise raise."""
 try:original.validate_declared_action(action,tools)
 except ValueError as error:
  if str(error).startswith(ARGUMENT_ERROR_PREFIXES):return str(error)
  raise
 return None

def refused_result(action,message):
 return {'ok':False,'status':ERROR_STATUS,'error':'ValueError','message':message,
  'arguments':deepcopy(action['arguments']),'host_invoked':False,'broker_invoked':False,'provider_invoked':False,
  'retry_authorized':False,'automatic_retry':False,'model_may_choose_new_call':True,'arguments_repaired':False}

class VisibleDeclaredArgumentModel(original.DeclaredHarmonySamplerModel):
 def parse(self,raw):
  try:return super().parse(raw)
  except ValueError as error:
   if not str(error).startswith(ARGUMENT_ERROR_PREFIXES):raise
   # The original native parser has already admitted channel/JSON/recipient.
   # Reconstruct only its exact original public action, never change a value.
   if self.segment is None or self.segment['kind']!='tool':raise
   names={'functions.'+name:name for name in original.validate_manifest(self.tool_definitions)}
   if self.segment['tool_name'] not in names:raise
   action={'kind':'tool','name':names[self.segment['tool_name']],'arguments':deepcopy(self.segment['arguments'])}
   if argument_error(action,self.tool_definitions)!=str(error):raise ValueError('EXACT_ARGUMENT_ERROR_SOURCE_REQUIRED')
   return action

 def observe(self,action,result):
  message=argument_error(action,self.tool_definitions)
  if message is None:return super().observe(action,result)
  if not exact_json_equal(result,refused_result(action,message)):
   raise ValueError('EXACT_ORIGINAL_ARGUMENT_ERROR_RESULT_REQUIRED')
  # Same complete-prefix append algorithm as the preserved original observe;
  # only the schema acceptance boundary differs for a visible refusal result.
  import tinker
  from tinker_cookbook.renderers.base import RenderContext
  from training import v7_protocol as protocol
  from training import direct_native_episode as native
  if (self.segment is None or self.segment['kind']!='tool'
   or self.segment['tool_name']!='functions.'+action['name']
   or not exact_json_equal(action['arguments'],self.segment['arguments'])):
   raise ValueError('exact native model tool arguments required')
  native.require(native.parse_direct_segment(self.segment['raw_tokens'],self.renderer,self.names)==self.segment,'DIRECT_TOOL_TRACE_CHANGED')
  content=canonical(result);protocol._safe_content(content,self.renderer)
  rendered=self.renderer.render_message({'role':'tool','name':self.segment['tool_name'],'content':content,
   'tool_call_id':'pure-native:'+str(len(self.training_steps))},RenderContext(idx=0,is_last=True))
  result_tokens=rendered.header.tokens+[token for chunk in rendered.output for token in chunk.tokens]
  suffix=self.renderer.tokenizer.encode('<|start|>assistant',add_special_tokens=False)
  native.require(self.prompt.to_ints()[-len(suffix):]==suffix,'DIRECT_NATIVE_PREFIX_CHANGED')
  tokens=self.prompt.to_ints()+self.segment['raw_tokens']+result_tokens+suffix
  native.require(len(tokens)<native.MAX_SEQUENCE,'DIRECT_CONTINUATION_CONTEXT_EXHAUSTED')
  self.prompt=tinker.ModelInput.from_ints(tokens)
  self._last_observation={'role':'tool','name':action['name'],'content':content};self.segment=None

class VisibleDeclaredArgumentRuntime(original.DeclaredAgentRuntime):
 """Same finite step/receipt ordering; argument refusals stay visible and count."""
 def step(self):
  if self.status not in {'READY','RUNNING'}:return self.report()
  if self.steps>=self.max_steps:self.status='STEP_LIMIT';return self.report()
  self.steps+=1;self.status='RUNNING';started=utc_now()
  try:
   raw=self.model.generate(deepcopy(self.conversation))
   if type(raw) is not bytes:raise TypeError('model backend must return original bytes')
   raw_ref=write_new(self.directory/f'step-{self.steps:04d}-model.raw',raw)
  except LocalGenerationBudgetExhausted as error:
   self.status='LOCAL_MODEL_BUDGET_EXHAUSTED'
   return self._receipt({'step':self.steps,'status':self.status,'error':type(error).__name__,'message':str(error),
    'details':error.details,'tool_invoked':False,'provider_dispatched':False,'started_at_utc':started})
  except Exception as error:
   self.status='MODEL_OR_RAW_CAPTURE_FAILED'
   return self._receipt({'step':self.steps,'status':self.status,'error':type(error).__name__,'message':str(error),
    'tool_invoked':False,'started_at_utc':started})
  try:
   action=self.model.parse(raw);message=argument_error(action,self.tool_definitions)
  except Exception as error:
   self.status='MALFORMED_OUTPUT'
   return self._receipt({'step':self.steps,'status':self.status,'error':type(error).__name__,'message':str(error),
    'raw_model_ref':raw_ref,'tool_invoked':False,'started_at_utc':started})
  if action['kind']=='final':
   self.status,self.final='FINAL',action['text'];self.conversation.append({'role':'assistant','content':action['text']})
   return self._receipt({'step':self.steps,'status':self.status,'action':action,'raw_model_ref':raw_ref,
    'tool_invoked':False,'started_at_utc':started})
  action_ref=write_new(self.directory/f'step-{self.steps:04d}-action.json',canonical(action).encode())
  if message is not None:
   result=refused_result(action,message);invoked=False
  else:
   invoked=True
   try:
    result=self.host.execute(deepcopy(action['arguments']))
    if type(result) is not dict:raise TypeError('host must return an object receipt')
    canonical(result)
   except Exception as error:
    result={'ok':False,'status':'HOST_EXCEPTION_OUTCOME_UNKNOWN','error':type(error).__name__,'message':str(error),
     'arguments':deepcopy(action['arguments']),'retry_authorized':False};self.status='HOST_EXCEPTION_OUTCOME_UNKNOWN'
  receipt=self._receipt({'step':self.steps,'status':self.status,'action':action,'raw_model_ref':raw_ref,'action_ref':action_ref,
   'tool_invoked':invoked,'local_argument_validation_refused':message is not None,'tool_result':deepcopy(result),
   'started_at_utc':started,'finished_at_utc':utc_now()})
  self.conversation+=[{'role':'assistant','tool_call':deepcopy(action)},
   {'role':'tool','name':action['name'],'content':canonical(result)}]
  self.model.observe(deepcopy(action),deepcopy(result))
  return receipt
