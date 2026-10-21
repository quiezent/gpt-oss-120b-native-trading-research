"""Authored synthetic accounting fixture, not a model or profitability test."""
from datetime import date,timedelta
import json
from training.nav_learning import PortfolioSimulationHost,SimulationConfig

def main():
    bars=[{'session':(date(2020,1,1)+timedelta(days=i)).isoformat(),'open':'100','high':'110','low':'90','close':'100','volume':'1000'} for i in range(80)]
    data={'SYNTH':{'symbol':'SYNTH','currency':'USD','contract_id':1,'synthetic_contract_identity':True,'primary_exchange':'SIMULATED','bars':bars}}
    h=PortfolioSimulationHost(data,{'start_session':bars[60]['session'],'horizon_sessions':5},SimulationConfig())
    # No investment action is prescribed; this fixture just marks the cash book.
    r=h.terminal_result()
    print(json.dumps({'synthetic':True,'initial_nav_usd':r['initial_nav_usd'],'terminal_nav_usd':r['terminal_nav_usd'],'reward':r['reward'],'forced_liquidations':r['forced_liquidations'],'provider_calls':0,'broker_calls':0}))
if __name__=='__main__':main()
