import os
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel,function_tool, set_tracing_disabled
from agents.run import RunConfig
import asyncio
from dotenv import load_dotenv
import requests
import rich

load_dotenv()
set_tracing_disabled(True)

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    raise ValueError("GEMINI_API_KEY is not present in .env file.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai"
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model_provider=external_client,
    model=model,
    tracing_disabled=True
)

@function_tool
async def get_current_location()-> str:
    """Returns the current location based on IP address."""
    try:
        api_key = os.getenv("LOCATION_API_KEY")
        url = f"https://ipinfo.io/json?token={api_key}"
        response = requests.get(url)
        data = response.json()
        return (
            f"Your current location is:\n"
            f"City: {data.get('city')}\n"
            f"Region: {data.get('region')}\n"
            f"Country: {data.get('country')}"
        )
    
    except Exception as e:
        return f"Error fetching location: {e}"

@function_tool
async def get_breaking_news()-> str:
    """Returns the latest breaking news.""" 
    try:
        api_key = os.getenv("BREAKING_NEWS_API_KEY")
        url = f"https://newsapi.org/v2/top-headlines?country=us&apiKey={api_key}"
        response = requests.get(url)
        articles = response.json().get("articles", [])
        
        if not articles:
            return "No breakig news available at the moment."
        
        latest_news = [f"- {a['title']}" for a in articles[:5]]
        return "\n".join(latest_news)
    
    except Exception as e:
        return f"Error fetching news: {e}"

plant_agent = Agent(
    name="PlantAgent",
    instructions="You are a plant biology expert who can explain photosynthesis clearly.",
    model=model
)

news_location_agent = Agent(
    name="NewsLocationAgent",
    instructions="You specialize in giving the user's current location and the latest breaking news, and can also answer questions about plant biology briefly.",
    model=model,
    tools=[get_current_location, get_breaking_news],
    handoffs=[plant_agent]
)

async def main():
    result = await Runner.run(
        starting_agent=news_location_agent,
        input=""" 
            1. What is my current location?
            3. Any breaking news?
            2. What is photosynthesis
        """,
        run_config=config
    )
    
    print('='*50)
    print("Result: ", result.last_agent.name)
    rich.print(result.new_items)
    print("Result: ",result.final_output)

if __name__ == "__main__":
    asyncio.run(main())