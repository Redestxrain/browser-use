"""
Goal: Searches for job listings, evaluates relevance based on a CV, and applies

@dev You need to add OPENAI_API_KEY to your environment variables.
Also you have to install PyPDF2 to read pdf files: pip install PyPDF2
"""

import csv
import os
import sys
from pathlib import Path
import logging
from typing import List, Optional
import asyncio


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from pydantic import BaseModel, SecretStr

from browser_use import ActionResult, Agent, Controller
from browser_use.browser.context import BrowserContext
from browser_use.browser.browser import Browser, BrowserConfig


def get_linkedin_credentials():
    """從環境變數安全讀取 LinkedIn 登入憑證，並使用 SecretStr 包裹密碼。"""
    email = os.getenv("linkedin_email")
    password = os.getenv("linkedin_password")
    if not email or not password:
        raise ValueError("linkedin_email 或 linkedin_password 未設定，請檢查環境變數。")
    return email, SecretStr(password)


# Validate required environment variables
load_dotenv()
# required_env_vars = ["AZURE_OPENAI_KEY", "AZURE_OPENAI_ENDPOINT"]
# for var in required_env_vars:
#     if not os.getenv(var):
#         raise ValueError(f"{var} is not set. Please add it to your environment variables.")

logger = logging.getLogger(__name__)
# full screen mode
controller = Controller()

# NOTE: This is the path to your cv file
CV = Path.cwd() / 'Hsu_Tse_Chun_s_CV_DS.pdf'

if not CV.exists():
    raise FileNotFoundError(f'You need to set the path to your cv file in the CV variable. CV file not found at {CV}')


class Job(BaseModel):
    title: str
    link: str
    company: str
    fit_score: float
    location: Optional[str] = None
    salary: Optional[str] = None


@controller.action('Save jobs to file - with a score how well it fits to my profile', param_model=Job)
def save_jobs(job: Job):
    with open('jobs.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([job.title, job.company, job.link, job.salary, job.location])

    return 'Saved job to file'


@controller.action('Read jobs from file')
def read_jobs():
    with open('jobs.csv', 'r') as f:
        return f.read()


@controller.action('LinkedIn login action')
async def linkedin_login(email: str, password: str, browser: BrowserContext):
    try:
        # Navigate to LinkedIn login page
        await browser.goto("https://www.linkedin.com/login")

        # Find email input field and type email
        email_input = await browser.wait_for_selector('input#username')
        await email_input.fill(email)

        # Find password input field and type password
        password_input = await browser.wait_for_selector('input#password')
        await password_input.fill(password)

        # Click the sign in button
        sign_in_button = await browser.wait_for_selector('button[type="submit"]')
        await sign_in_button.click()

        # Wait for navigation to complete after login
        await browser.wait_for_navigation()

        msg = "Successfully logged in to LinkedIn"
        logger.info(msg)
        return ActionResult(extracted_content=msg, include_in_memory=True)

    except Exception as e:
        error_msg = f"Failed to login to LinkedIn: {str(e)}"
        logger.error(error_msg)
        return ActionResult(error=error_msg)

@controller.action('Read my cv for context to fill forms')
def read_cv():
    pdf = PdfReader(CV)
    text = ''
    for page in pdf.pages:
        text += page.extract_text() or ''
    logger.info(f'Read cv with {len(text)} characters')
    return ActionResult(extracted_content=text, include_in_memory=True)


@controller.action(
    'Upload cv to element - call this function to upload if element is not found, try with different index of the same upload element',
)
async def upload_cv(index: int, browser: BrowserContext):
    path = str(CV.absolute())
    dom_el = await browser.get_dom_element_by_index(index)

    if dom_el is None:
        return ActionResult(error=f'No element found at index {index}')

    file_upload_dom_el = dom_el.get_file_upload_element()

    if file_upload_dom_el is None:
        logger.info(f'No file upload element found at index {index}')
        return ActionResult(error=f'No file upload element found at index {index}')

    file_upload_el = await browser.get_locate_element(file_upload_dom_el)

    if file_upload_el is None:
        logger.info(f'No file upload element found at index {index}')
        return ActionResult(error=f'No file upload element found at index {index}')

    try:
        await file_upload_el.set_input_files(path)
        msg = f'Successfully uploaded file "{path}" to index {index}'
        logger.info(msg)
        return ActionResult(extracted_content=msg)
    except Exception as e:
        logger.debug(f'Error in set_input_files: {str(e)}')
        return ActionResult(error=f'Failed to upload file to index {index}')


browser = Browser(
    config=BrowserConfig(
        disable_security=True,
    )
)


async def main():
    # ground_task = (
    # 	'You are a professional job finder. '
    # 	'1. Read my cv with read_cv'
    # 	'2. Read the saved jobs file '
    # 	'3. start applying to the first link of Amazon '
    # 	'You can navigate through pages e.g. by scrolling '
    # 	'Make sure to be on the english version of the page'
    # )
    ground_task = (
        'You are a professional job finder. '
        '1. Go to www.linkedin.com/login and use the LinkedIn login action with my credentials. '
        '2. Read my cv with read_cv. '
        '3. Find an internships and apply them and save them to a file. '
        '4. need to upload my cv to the application form. '
        '5. You can search at this role and apply all of them in the list.'
        'search at this role:'

    )

    tasks = [
        ground_task + '\n' + 'data analyst intern',
        ground_task + '\n' + 'data scientist intern',
    ]
    model = ChatOpenAI(
        model='gpt-4o',
        temperature=0.0,
    )

    agents = []
    for task in tasks:
        agent = Agent(task=task, llm=model, controller=controller, browser=browser)
        agents.append(agent)

    await asyncio.gather(*[agent.run() for agent in agents])


if __name__ == "__main__":
    asyncio.run(main())
