from langgraph.graph import StateGraph, END

def dummy_node(state):
    print("Node executed!")
    return state

graph = StateGraph(dict)
graph.add_node("start", dummy_node)
graph.set_entry_point("start")
graph.set_finish_point("start")

app = graph.compile()

app.invoke({})
