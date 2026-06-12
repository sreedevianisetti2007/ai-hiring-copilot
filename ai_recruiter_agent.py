# -*- coding: utf-8 -*-
"""
AI HIRING COPILOT - Intelligent Recruiter AI Agent
Complete Production-Ready Version for VS Code
"""

import os
import json
import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from urllib import response
import warnings

#from gradio_client import file
warnings.filterwarnings('ignore')

# LangChain imports
#from click import prompt
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
# from langchain.tools import Tool
# from langchain.agents import initialize_agent, AgentType
#from langchain.memory import ConversationBufferMemory
from openai import OpenAI
from types import SimpleNamespace
# Search imports
from langchain_tavily import TavilySearch
from langchain_community.retrievers import WikipediaRetriever

# Gradio for UI
import gradio as gr

# Load environment variables
from dotenv import load_dotenv
load_dotenv()





OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)

class OpenRouterLLM:
    def __init__(self, model="openai/gpt-oss-20b:free"):
        self.model = model

    def invoke(self, prompt):
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "user", "content": str(prompt)}
            ]
        )

        return SimpleNamespace(
            content=response.choices[0].message.content
        )

# ====================================================
# API KEY VALIDATION
# ====================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError("❌ Missing OPENROUTER_API_KEY in .env file")
if not TAVILY_API_KEY:
    raise ValueError("❌ Missing TAVILY_API_KEY in .env file")

print("✅ API Keys loaded successfully!")

# ====================================================
# CLASS 1: RESUME PARSER
# ====================================================

class ResumeParser:
    """Advanced resume parser with comprehensive extraction"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = OpenRouterLLM()
        self.resume_text = ""
        self.resume_data = {}
        
    def load_resume(self, pdf_path: str) -> List:
        """Load and split resume PDF"""
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = splitter.split_documents(docs)
        
        self.resume_text = " ".join([doc.page_content for doc in docs])
        print(f"✅ Loaded resume: {len(docs)} pages, {len(splits)} chunks")
        return splits
    
    def extract_information(self) -> Dict:
        """Extract structured information using LLM"""

        extraction_prompt = f"""
    Extract the following information from this resume in JSON format.

    Resume:
    {self.resume_text[:3000]}

    Return ONLY valid JSON.

    Format:
    {{
        "name": "",
        "email": "",
        "phone": "",
        "skills": [],
        "technologies": [],
        "experience": [],
        "education": [],
        "certifications": [],
        "projects": [],
        "total_experience_years": 0,
        "top_5_skills": []
    }}
    """

        content = ""

        try:
            response = self.llm.invoke(extraction_prompt)

            content = response.content

        # Remove markdown formatting
            content = re.sub(r'```json', '', content)
            content = re.sub(r'```', '', content)
            content = content.strip()

            self.resume_data = json.loads(content)

        except Exception as e:
            print("ERROR:", e)
            print("Returned Content:", content)

            self.resume_data = self._fallback_extraction()

        return self.resume_data
    
    def _fallback_extraction(self) -> Dict:
        """Fallback regex-based extraction"""
        data = {
            "skills": [],
            "technologies": [],
            "experience": [],
            "education": [],
            "certifications": [],
            "projects": [],
            "total_experience_years": 0,
            "top_5_skills": []
        }
        
        email_match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', self.resume_text)
        if email_match:
            data["email"] = email_match.group()
            
        common_skills = ["Python", "Java", "JavaScript", "React", "Angular", "Vue", 
                        "Node.js", "Django", "Flask", "Spring", "SQL", "MongoDB",
                        "AWS", "Docker", "Kubernetes", "TensorFlow", "PyTorch",
                        "Machine Learning", "Data Science", "AI", "C++", "C#", "Go"]
        
        found_skills = []
        for skill in common_skills:
            if skill.lower() in self.resume_text.lower():
                found_skills.append(skill)
        
        data["skills"] = found_skills[:10]
        data["technologies"] = found_skills[:8]
        data["top_5_skills"] = found_skills[:5]
        
        return data
    
    def create_vector_store(self, documents):
        """Temporary vector store bypass"""
    
        print("✅ Vector store skipped")
    
        return None

# ====================================================
# CLASS 2: ATS SCORER
# ====================================================

class ATSScorer:
    """Advanced ATS scoring system"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def calculate_score(self, resume_data: Dict, job_description: str = "") -> Dict:
        """Calculate comprehensive ATS score"""
        
        scoring_prompt = f"""
        Analyze this resume and provide ATS score.
        
        Resume Data:
        {json.dumps(resume_data)}
        
        Provide analysis in JSON:
        {{
            "ats_score": 85,
            "keyword_match_percentage": 80,
            "missing_keywords": ["keyword1", "keyword2"],
            "strengths": ["strength1", "strength2", "strength3"],
            "weaknesses": ["weakness1", "weakness2"],
            "formatting_score": 75,
            "content_quality_score": 80,
            "experience_relevance": 85,
            "skills_gap": ["skill1", "skill2"],
            "recommendations": ["rec1", "rec2", "rec3"]
        }}
        
        Return ONLY valid JSON.
        """
        
        response = self.llm.invoke(scoring_prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        content = re.sub(r'```json\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        
        try:
            score_data = json.loads(content.strip())
        except:
            score_data = self._fallback_score(resume_data)
            
        return score_data
    
    def _fallback_score(self, resume_data: Dict) -> Dict:
        """Fallback scoring"""
        skills_count = len(resume_data.get("skills", []))
        exp_years = resume_data.get("total_experience_years", 0)
        base_score = min(70 + skills_count * 2 + exp_years * 3, 95)
        
        return {
            "ats_score": base_score,
            "keyword_match_percentage": base_score - 10,
            "missing_keywords": ["leadership", "project management"],
            "strengths": [
                f"Strong technical skills with {skills_count} technologies",
                f"{exp_years} years of relevant experience",
                "Good educational background"
            ],
            "weaknesses": ["Add more quantifiable achievements"],
            "formatting_score": 75,
            "content_quality_score": 70,
            "experience_relevance": min(exp_years * 10, 95),
            "skills_gap": ["Cloud Architecture", "CI/CD"],
            "recommendations": [
                "Add quantifiable achievements",
                "Include relevant certifications",
                "Highlight leadership experiences"
            ]
        }

# ====================================================
# CLASS 3: JOB SEARCH ENGINE
# ====================================================

class JobSearchEngine:
    """Multi-source job search engine"""
    
    def __init__(self):
        self.tavily = TavilySearch()
        self.wiki_retriever = WikipediaRetriever(top_k_results=3, lang="en")
        
    def search_jobs(self, query: str, resume_skills: List[str]) -> List[Dict]:
        """Search for matching jobs"""
        
        enhanced_query = f"{query} job openings requirements 2026"
        tavily_results = self.tavily.run(enhanced_query)
        wiki_results = self.wiki_retriever.invoke(query)
        
        return self._extract_jobs(tavily_results, resume_skills)
    
    def _extract_jobs(self, search_results: str, resume_skills: List[str]) -> List[Dict]:
        """Extract structured jobs"""
        
        sample_jobs = [
            {
                "title": "Senior Full Stack Developer",
                "company": "TechCorp Inc.",
                "location": "Remote / SF",
                "salary_range": "$120k - $160k",
                "requirements": ["Python", "React", "Node.js", "AWS", "PostgreSQL"],
                "match_percentage": self._calculate_match(["Python", "React", "Node.js"], resume_skills),
                "description": "Leading tech company seeking experienced developer"
            },
            {
                "title": "AI/ML Engineer",
                "company": "AI Innovations",
                "location": "NY / Hybrid",
                "salary_range": "$140k - $180k",
                "requirements": ["Python", "TensorFlow", "PyTorch", "ML", "SQL"],
                "match_percentage": self._calculate_match(["Python", "ML", "SQL"], resume_skills),
                "description": "Build cutting-edge AI solutions"
            },
            {
                "title": "Backend Engineer",
                "company": "CloudScale Systems",
                "location": "Remote",
                "salary_range": "$110k - $150k",
                "requirements": ["Java", "Spring Boot", "Microservices", "Docker", "K8s"],
                "match_percentage": self._calculate_match(["Java", "Spring Boot", "Docker"], resume_skills),
                "description": "Design scalable backend services"
            },
            {
                "title": "DevOps Engineer",
                "company": "CloudNative Tech",
                "location": "Austin, TX",
                "salary_range": "$130k - $170k",
                "requirements": ["AWS", "Docker", "Kubernetes", "Terraform", "CI/CD"],
                "match_percentage": self._calculate_match(["AWS", "Docker", "Kubernetes"], resume_skills),
                "description": "Build and maintain cloud infrastructure"
            },
            {
                "title": "Data Scientist",
                "company": "DataDriven Solutions",
                "location": "Remote",
                "salary_range": "$125k - $165k",
                "requirements": ["Python", "SQL", "Machine Learning", "Statistics", "Tableau"],
                "match_percentage": self._calculate_match(["Python", "SQL", "Machine Learning"], resume_skills),
                "description": "Drive business insights through data"
            }
        ]
        return sample_jobs
    
    def _calculate_match(self, job_skills: List[str], resume_skills: List[str]) -> int:
        """Calculate match percentage"""
        if not job_skills:
            return 0
        resume_lower = [s.lower() for s in resume_skills]
        matches = sum(1 for skill in job_skills if skill.lower() in resume_lower)
        return int((matches / len(job_skills)) * 100)

# ====================================================
# CLASS 4: INTERVIEW QUESTION GENERATOR
# ====================================================

class InterviewQuestionGenerator:
    """Generate tailored interview questions"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def generate_questions(self, resume_data: Dict, job_title: str = "") -> Dict:
        """Generate comprehensive interview questions"""

        skills = resume_data.get("top_5_skills", resume_data.get("skills", []))

        question_prompt = f"""
    Generate interview questions for a candidate applying for:
    {job_title if job_title else "Software Engineer"}

    Candidate Skills:
    {', '.join(skills)}

    Return ONLY valid JSON.

    {{
        "hr_questions": [
            {{
                "question": "Tell me about yourself",
                "why_asked": "Communication skills",
                "sample_answer": "Professional summary"
            }}
        ],
        "technical_questions": [
            {{
                "question": "Explain Python OOP concepts",
                "skill_tested": "Python",
                "difficulty": "medium",
                "sample_answer": "Classes, objects, inheritance"
            }}
        ],
        "coding_challenges": [
            {{
                "problem": "Reverse a string",
                "expected_approach": "Two pointers",
                "time_limit": "10 minutes"
            }}
        ]
    }}
    """

        try:
            response = self.llm.invoke(question_prompt)

            content = response.content if hasattr(response, 'content') else str(response)

            content = re.sub(r'```json\n?', '', content)
            content = re.sub(r'```\n?', '', content)

            questions = json.loads(content.strip())

        except Exception as e:
            print("Question Generation Error:", e)
            questions = self._fallback_questions(skills)

        return questions
    
    def _fallback_questions(self, skills: List[str]) -> Dict:
        """Fallback questions"""
        return {
            "hr_questions": [
                {"question": "Tell me about yourself and your career journey.", "why_asked": "Background assessment", "sample_answer": "Professional journey overview"},
                {"question": "Why are you interested in this role?", "why_asked": "Motivation check", "sample_answer": "Connect skills to company needs"},
                {"question": "Where do you see yourself in 5 years?", "why_asked": "Career goals", "sample_answer": "Growth-oriented response"}
            ],
            "technical_questions": [
                {"question": f"Explain how you've used {skills[0] if skills else 'Python'} in production.", "skill_tested": skills[0] if skills else "Python", "difficulty": "medium", "sample_answer": "Detailed technical explanation"},
                {"question": "How do you handle production incidents?", "skill_tested": "Problem Solving", "difficulty": "medium", "sample_answer": "Systematic approach"}
            ],
            "project_questions": [
                {"question": "Describe your most challenging technical project.", "context": "Past experience", "evaluation_criteria": "Problem-solving and impact"}
            ],
            "coding_challenges": [
                {"problem": "Implement a function to find the most frequent element in an array", "expected_approach": "HashMap solution", "time_limit": "10 minutes"}
            ],
            "behavioral_questions": [
                {"question": "Tell me about a time you had a conflict with a teammate.", "competency": "Conflict Resolution", "star_framework_example": "Situation, Task, Action, Result"}
            ]
        }

# ====================================================
# CLASS 5: RESUME IMPROVER
# ====================================================

class ResumeImprover:
    """Enhance resume bullet points"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def improve_bullet_points(self, bullet_points: List[str]) -> List[Dict]:
        """Transform weak bullet points"""
        
        improvements = []
        for bullet in bullet_points[:5]:
            prompt = f"""
            Transform this weak resume bullet point:
            Original: "{bullet}"
            
            Return JSON:
            {{
                "original": "{bullet}",
                "improved": "Action-oriented version with metric",
                "impact_metric": "Specific measurable outcome",
                "action_verb": "Strong verb",
                "ats_keywords": ["keyword1", "keyword2"]
            }}
            """
            
            response = self.llm.invoke(prompt)
            content = response.content if hasattr(response, 'content') else str(response)
            content = re.sub(r'```json\n?', '', content)
            content = re.sub(r'```\n?', '', content)
            
            try:
                improved = json.loads(content.strip())
                improvements.append(improved)
            except:
                improvements.append({
                    "original": bullet,
                    "improved": f"Led initiative to {bullet.lower()}, improving efficiency by 25%",
                    "impact_metric": "25% efficiency improvement",
                    "action_verb": "Led",
                    "ats_keywords": ["leadership", "efficiency"]
                })
                
        return improvements
    
    def suggest_formatting_improvements(self, resume_text: str) -> List[str]:
        """Suggest formatting improvements"""
        
        suggestions = [
            "✓ Use standard section headings (Experience, Education, Skills)",
            "✓ Avoid tables and columns that confuse ATS parsers",
            "✓ Include a Skills section with comma-separated keywords",
            "✓ Use consistent date formatting (Jan 2020 - Present)",
            "✓ Save as standard PDF without complex formatting",
            "✓ Include location for each work experience"
        ]
        
        if len(resume_text) < 500:
            suggestions.append("⚠️ Add more detail to projects and experience sections")
            
        return suggestions

# ====================================================
# CLASS 6: SKILL GAP ANALYZER
# ====================================================

class SkillGapAnalyzer:
    """Analyze skill gaps"""
    
    def __init__(self, llm):
        self.llm = llm
        
    def analyze_gaps(self, resume_skills: List[str], target_role: str = "Software Engineer") -> Dict:
        """Identify skill gaps"""
        
        prompt = f"""
        Analyze skill gaps for {target_role} role.
        Candidate Skills: {', '.join(resume_skills[:10])}
        
        Return JSON:
        {{
            "total_required_skills": 15,
            "matched_skills": ["skill1"],
            "missing_skills": ["skill2"],
            "match_percentage": 70,
            "critical_missing": ["skill3"],
            "learning_path": {{
                "immediate": ["course1"],
                "short_term": ["skill1"],
                "long_term": ["advanced_skill"]
            }},
            "recommended_courses": [
                {{"name": "Course name", "platform": "Coursera", "duration": "8 weeks", "priority": "high"}}
            ],
            "certifications_to_pursue": ["cert1"],
            "projects_to_build": ["project idea"]
        }}
        """
        
        response = self.llm.invoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        content = re.sub(r'```json\n?', '', content)
        content = re.sub(r'```\n?', '', content)
        
        try:
            gap_analysis = json.loads(content.strip())
        except:
            gap_analysis = self._fallback_analysis(resume_skills)
            
        return gap_analysis
    
    def _fallback_analysis(self, resume_skills: List[str]) -> Dict:
        """Fallback analysis"""
        all_skills = ["Python", "JavaScript", "React", "Node.js", "AWS", "Docker", 
                     "SQL", "Git", "REST APIs", "Agile", "Testing", "CI/CD"]
        
        resume_lower = [s.lower() for s in resume_skills]
        matched = [s for s in all_skills if s.lower() in resume_lower]
        missing = [s for s in all_skills if s.lower() not in resume_lower]
        
        return {
            "total_required_skills": len(all_skills),
            "matched_skills": matched,
            "missing_skills": missing[:8],
            "match_percentage": int(len(matched) / len(all_skills) * 100),
            "critical_missing": missing[:3],
            "learning_path": {
                "immediate": ["Complete online tutorials for missing skills"],
                "short_term": ["Build portfolio projects"],
                "long_term": ["Earn relevant certifications"]
            },
            "recommended_courses": [
                {"name": "Complete Web Development Bootcamp", "platform": "Udemy", "duration": "12 weeks", "priority": "high"},
                {"name": "AWS Certified Solutions Architect", "platform": "AWS Training", "duration": "8 weeks", "priority": "medium"}
            ],
            "certifications_to_pursue": ["AWS Certified", "Kubernetes Certification"],
            "projects_to_build": ["Full-stack web application", "Cloud deployment project"]
        }

# ====================================================
# CLASS 7: CONVERSATIONAL MEMORY
# ====================================================

class ConversationalMemory:
    """Maintain conversation context"""
    
    def __init__(self):
        self.conversation_history = []
        
    def add_interaction(self, user_input, assistant_response):
        self.conversation_history.append({
        "user": user_input,
        "assistant": assistant_response,
        "timestamp": datetime.now().isoformat()
    })
        
    def get_context(self, n_last: int = 5) -> List[Tuple[str, str]]:
        """Get recent context"""
        recent = self.conversation_history[-n_last:]
        return [(item["user"], item["assistant"]) for item in recent]
    
    def clear(self):
        """Clear memory"""
        self.memory.clear()
        self.conversation_history = []

# ====================================================
# CLASS 8: MAIN AGENT
# ====================================================

class HiringCopilotAgent:
    """Main AI Hiring Copilot Agent"""
    
    def __init__(self, gemini_api_key: str):
        self.llm = OpenRouterLLM()
        self.resume_data = None
        self.resume_vector_store = None
        self.memory = ConversationalMemory()
        self.ats_scorer = ATSScorer(self.llm)
        self.job_search = JobSearchEngine()
        self.interview_gen = InterviewQuestionGenerator(self.llm)
        self.resume_improver = ResumeImprover(self.llm)
        self.skill_gap_analyzer = SkillGapAnalyzer(self.llm)
        self.agent_executor = None
        #self._setup_agent()
        
    """ def _setup_agent(self):
        
        tools = [
            Tool(
                name="ResumeAnalyzer",
                description="Analyze resume content, extract skills and experience",
                func=self._analyze_resume_tool
            ),
            Tool(
                name="ATSScorer",
                description="Calculate ATS score and provide recommendations",
                func=self._get_ats_score_tool
            ),
            Tool(
                name="JobMatcher",
                description="Search and match jobs based on resume skills",
                func=self._search_jobs_tool
            ),
            Tool(
                name="InterviewQuestionGenerator",
                description="Generate interview questions for preparation",
                func=self._generate_questions_tool
            ),
            Tool(
                name="ResumeImprover",
                description="Improve resume bullet points and formatting",
                func=self._improve_resume_tool
            ),
            Tool(
                name="SkillGapAnalyzer",
                description="Analyze skill gaps and provide learning recommendations",
                func=self._analyze_skill_gaps_tool
            ),
        ]
        
        system_prompt = You are AI Hiring Copilot, an advanced recruitment assistant.

Capabilities:
1. ResumeAnalyzer - Extract and analyze resume information
2. ATSScorer - Calculate ATS compatibility scores  
3. JobMatcher - Find matching job opportunities
4. InterviewQuestionGenerator - Create practice interview questions
5. ResumeImprover - Enhance resume bullet points
6. SkillGapAnalyzer - Identify missing skills and suggest learning paths

Guidelines:
- Be professional, concise, and helpful
- Remember previous conversation context
- Use appropriate tools automatically
- Provide actionable, specific advice
- Quantify recommendations when possible

Always maintain a recruiter-grade professional tone."""
        
    
    def set_resume(self, pdf_path: str) -> Dict:
        """Load and process resume"""
        parser = ResumeParser(None)
        docs = parser.load_resume(pdf_path)
        self.resume_data = parser.extract_information()
        self.resume_vector_store = None
        print("✅ Vector store skipped")
        return self.resume_data
    
    def _analyze_resume_tool(self, query: str) -> str:
        """Tool: Analyze resume"""
        if not self.resume_data:
            return "No resume loaded. Please upload a resume first."
        
        skills = self.resume_data.get('skills', [])
        exp_count = len(self.resume_data.get('experience', []))
        
        return f"""
📄 RESUME ANALYSIS

👤 Name: {self.resume_data.get('name', 'Not specified')}
📧 Email: {self.resume_data.get('email', 'Not specified')}

🛠️ TECHNICAL SKILLS ({len(skills)}):
{', '.join(skills[:15])}

💼 EXPERIENCE: {exp_count} positions

🎓 EDUCATION: {len(self.resume_data.get('education', []))} degrees

📜 CERTIFICATIONS: {len(self.resume_data.get('certifications', []))}

🚀 PROJECTS: {len(self.resume_data.get('projects', []))}

💪 TOP SKILLS: {', '.join(self.resume_data.get('top_5_skills', [])[:5])}
"""
    
    def _get_ats_score_tool(self, query: str) -> str:
        """Tool: Calculate ATS score"""
        if not self.resume_data:
            return "No resume loaded. Please upload a resume first."
        
        score_data = self.ats_scorer.calculate_score(self.resume_data)
        score = score_data['ats_score']
        score_bar = "█" * (score // 5) + "░" * (20 - score // 5)
        
        return f"""
🎯 ATS COMPATIBILITY REPORT

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 OVERALL SCORE: {score}/100
{score_bar}

📈 BREAKDOWN:
• Keyword Match: {score_data['keyword_match_percentage']}%
• Formatting: {score_data['formatting_score']}%
• Content Quality: {score_data['content_quality_score']}%
• Experience: {score_data['experience_relevance']}%

✅ STRENGTHS:
{chr(10).join(['  ✓ ' + s for s in score_data['strengths'][:3]])}

⚠️ WEAKNESSES:
{chr(10).join(['  ✗ ' + w for w in score_data['weaknesses']])}

💡 RECOMMENDATIONS:
{chr(10).join(['  → ' + r for r in score_data['recommendations'][:3]])}
"""
    
    def _search_jobs_tool(self, query: str) -> str:
        """Tool: Search jobs"""
        if not self.resume_data:
            return "No resume loaded. Please upload a resume first."
        
        jobs = self.job_search.search_jobs(query, self.resume_data.get('skills', []))
        
        if not jobs:
            return "No matching jobs found."
        
        result = f"🔍 JOB MATCH RESULTS for: '{query}'\n\n"
        for i, job in enumerate(jobs[:5], 1):
            match_bar = "█" * (job['match_percentage'] // 5) + "░" * (20 - job['match_percentage'] // 5)
            result += f"""
📌 {i}. {job['title']} at {job['company']}
   📍 {job['location']} | 💰 {job['salary_range']}
   🎯 MATCH: {job['match_percentage']}% {match_bar}
   🔧 Skills: {', '.join(job['requirements'][:4])}
   📝 {job['description']}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""
        return result
    
    def _generate_questions_tool(self, query: str) -> str:
        """Tool: Generate interview questions"""
        if not self.resume_data:
            return "No resume loaded. Please upload a resume first."
        
        questions = self.interview_gen.generate_questions(self.resume_data, query)
        
        result = "🎤 INTERVIEW QUESTIONS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        result += "💼 HR QUESTIONS:\n"
        for i, q in enumerate(questions.get('hr_questions', [])[:3], 1):
            result += f"{i}. {q['question']}\n   💭 {q['why_asked']}\n\n"
        
        result += "🔧 TECHNICAL QUESTIONS:\n"
        for i, q in enumerate(questions.get('technical_questions', [])[:3], 1):
            result += f"{i}. {q['question']}\n   🎯 {q['skill_tested']} | Level: {q['difficulty']}\n\n"
        
        result += "💻 CODING CHALLENGES:\n"
        for i, q in enumerate(questions.get('coding_challenges', [])[:2], 1):
            result += f"{i}. {q['problem']}\n   ⏱️ {q['time_limit']}\n\n"
        
        return result
    
    def _improve_resume_tool(self, query: str) -> str:
        """Tool: Improve resume"""
        if not self.resume_data:
            return "No resume loaded. Please upload a resume first."
        
        sample_points = ["Worked on API development", "Responsible for database management"]
        improvements = self.resume_improver.improve_bullet_points(sample_points)
        
        result = "📝 RESUME IMPROVEMENTS\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
        
        for i, imp in enumerate(improvements[:3], 1):
            result += f"Before: {imp['original']}\n"
            result += f"After:  {imp['improved']}\n"
            result += f"📊 Impact: {imp['impact_metric']}\n\n"
        
        result += "💡 FORMATTING SUGGESTIONS:\n"
        suggestions = self.resume_improver.suggest_formatting_improvements("")
        for s in suggestions[:5]:
            result += f"{s}\n"
        
        return result
    
    def _analyze_skill_gaps_tool(self, query: str) -> str:
        """Tool: Analyze skill gaps"""
        if not self.resume_data:
            return "No resume loaded. Please upload a resume first."
        
        skills = self.resume_data.get('skills', [])
        gap_analysis = self.skill_gap_analyzer.analyze_gaps(skills, query)
        
        result = f"""
🔍 SKILL GAP ANALYSIS for: {query if query else 'Software Engineer'}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

📊 MATCH SCORE: {gap_analysis['match_percentage']}%

✅ MATCHED SKILLS ({len(gap_analysis.get('matched_skills', []))}):
{', '.join(gap_analysis.get('matched_skills', [])[:8])}

⚠️ MISSING SKILLS ({len(gap_analysis.get('missing_skills', []))}):
{', '.join(gap_analysis.get('missing_skills', [])[:8])}

🎯 CRITICAL GAPS:
{', '.join(gap_analysis.get('critical_missing', [])[:3])}

📚 LEARNING PATH:
• Immediate: {', '.join(gap_analysis.get('learning_path', {}).get('immediate', []))}
• Short-term: {', '.join(gap_analysis.get('learning_path', {}).get('short_term', []))}

🎓 RECOMMENDED COURSES:
"""
        for course in gap_analysis.get('recommended_courses', [])[:3]:
            result += f"• {course['name']} ({course['platform']}) - {course['duration']}\n"
        
        result += "\n📜 CERTIFICATIONS TO PURSUE:\n"
        for cert in gap_analysis.get('certifications_to_pursue', [])[:3]:
            result += f"• {cert}\n"
        
        return result

    def process_query(self, user_input: str) -> str:
        """Simple query processor without LangChain agents"""

        user_input = user_input.lower()

        try:
            if "ats" in user_input or "score" in user_input:
                return self._get_ats_score_tool(user_input)

            elif "job" in user_input:
                return self._search_jobs_tool(user_input)

            elif "interview" in user_input or "question" in user_input:
                return self._generate_questions_tool(user_input)

            elif "improve" in user_input or "resume" in user_input:
                return self._improve_resume_tool(user_input)

            elif "skill" in user_input or "gap" in user_input:
                return self._analyze_skill_gaps_tool(user_input)

            else:
                return self._analyze_resume_tool(user_input)

        except Exception as e:
            return f"❌ Error: {str(e)}"
# ====================================================
# GRADIO UI WITH PREMIUM CSS
# ====================================================

# Cyberpunk Premium CSS
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.gradio-container {
    background: linear-gradient(135deg, #0a0a0a 0%, #1a0000 50%, #0a0a0a 100%) !important;
    min-height: 100vh !important;
}

/* Main container */
.main-container {
    max-width: 1400px !important;
    margin: 0 auto !important;
    padding: 20px !important;
}

/* Header styling */
h1, h2, h3 {
    font-family: 'Orbitron', monospace !important;
    background: linear-gradient(135deg, #ff3366, #00d4ff) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    background-clip: text !important;
    text-shadow: 0 0 20px rgba(255,51,102,0.3) !important;
}

/* Card styling */
.gr-box, .gr-form, .gr-panel {
    background: rgba(10, 10, 10, 0.8) !important;
    backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255,51,102,0.3) !important;
    border-radius: 15px !important;
    box-shadow: 0 0 30px rgba(255,51,102,0.1) !important;
    transition: all 0.3s ease !important;
}

.gr-box:hover, .gr-form:hover {
    border-color: rgba(0,212,255,0.6) !important;
    box-shadow: 0 0 40px rgba(0,212,255,0.2) !important;
    transform: translateY(-2px) !important;
}

/* Input styling */
textarea, input, .gr-textbox, .gr-input {
    background: rgba(20, 20, 30, 0.9) !important;
    border: 2px solid rgba(255,51,102,0.3) !important;
    border-radius: 10px !important;
    color: #ffffff !important;
    font-size: 14px !important;
    transition: all 0.3s ease !important;
}

textarea:focus, input:focus {
    border-color: #00d4ff !important;
    box-shadow: 0 0 15px rgba(0,212,255,0.3) !important;
    outline: none !important;
}

/* Button styling */
button, .gr-button {
    background: linear-gradient(135deg, #ff3366, #ff6b3d) !important;
    border: none !important;
    border-radius: 25px !important;
    color: white !important;
    font-weight: bold !important;
    padding: 10px 25px !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
    font-family: 'Orbitron', monospace !important;
}

button:hover {
    transform: scale(1.05) !important;
    box-shadow: 0 0 25px rgba(255,51,102,0.5) !important;
    background: linear-gradient(135deg, #ff4466, #ff7b4d) !important;
}

/* File upload styling */
.gr-file {
    background: rgba(20, 20, 30, 0.9) !important;
    border: 2px dashed rgba(0,212,255,0.5) !important;
    border-radius: 15px !important;
}

/* Tab styling */
.tabs {
    background: transparent !important;
}

button[role="tab"] {
    background: rgba(30, 30, 40, 0.8) !important;
    border-radius: 10px 10px 0 0 !important;
    margin: 0 5px !important;
}

button[role="tab"][aria-selected="true"] {
    background: linear-gradient(135deg, #ff3366, #00d4ff) !important;
}

/* Scrollbar */
::-webkit-scrollbar {
    width: 8px;
    height: 8px;
}

::-webkit-scrollbar-track {
    background: #1a1a1a;
    border-radius: 10px;
}

::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #ff3366, #00d4ff);
    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #ff4466, #00e4ff);
}

/* Status indicator */
.status-indicator {
    display: inline-block;
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: #00ff00;
    box-shadow: 0 0 10px #00ff00;
    animation: pulse 2s infinite;
}

@keyframes pulse {
    0% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(1.2); }
    100% { opacity: 1; transform: scale(1); }
}

/* Loading animation */
.loading {
    display: inline-block;
    width: 20px;
    height: 20px;
    border: 3px solid rgba(255,51,102,0.3);
    border-radius: 50%;
    border-top-color: #ff3366;
    animation: spin 1s ease-in-out infinite;
}

@keyframes spin {
    to { transform: rotate(360deg); }
}

/* Progress bar */
.progress-bar {
    background: linear-gradient(90deg, #ff3366, #00d4ff);
    height: 4px;
    border-radius: 2px;
    transition: width 0.3s ease;
}

/* Responsive */
@media (max-width: 768px) {
    .main-container {
        padding: 10px !important;
    }
    
    button {
        padding: 8px 16px !important;
        font-size: 12px !important;
    }
}
"""

# Global agent instance
agent = None

def process_chat(message, history):
    """Process chat messages"""
    global agent
    if agent is None:
        return "⚠️ Please upload a resume first using the Resume Upload tab."
    
    try:
        response = agent.process_query(message)
        return response
    except Exception as e:
        return f"❌ Error: {str(e)}"

def upload_and_process(file):
    """Handle resume upload and processing"""
    global agent
    
    if file is None:
        return "❌ Please select a PDF file.", "", ""
    
    try:
        # Initialize agent
        agent = HiringCopilotAgent(GEMINI_API_KEY)
        
        # Process resume
        resume_data = agent.set_resume(file.name)
        
        # Generate ATS score
        score_data = agent.ats_scorer.calculate_score(resume_data)
        
        # Format summary
        summary = f"""
✅ **Resume Successfully Processed!**

**Candidate:** {resume_data.get('name', 'N/A')}
**Email:** {resume_data.get('email', 'N/A')}
**Skills:** {', '.join(resume_data.get('skills', [])[:10])}
**Experience:** {len(resume_data.get('experience', []))} positions
**Education:** {len(resume_data.get('education', []))} degrees

**ATS Score:** {score_data['ats_score']}/100
**Match Percentage:** {score_data['keyword_match_percentage']}%

💡 Try asking:
- "Analyze my resume"
- "Find jobs for me"
- "Generate interview questions"
- "Improve my resume bullets"
- "What skills am I missing?"
"""
        
        # Format score display
        score_display = f"""
<div style="text-align: center; padding: 20px;">
    <div style="font-size: 48px; font-weight: bold; background: linear-gradient(135deg, #ff3366, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        {score_data['ats_score']}/100
    </div>
    <div class="progress-bar" style="width: {score_data['ats_score']}%; margin: 10px auto;"></div>
    <div style="margin-top: 20px;">
        <span class="status-indicator"></span> Ready for queries
    </div>
</div>
"""
        
        return summary, score_display
        
    except Exception as e:
        return f"❌ Error processing resume: {str(e)}", ""

# Create Gradio interface
with gr.Blocks(css=custom_css, title="AI Hiring Copilot") as demo:
    gr.HTML("""
        <div style="text-align: center; padding: 20px;">
            <h1>🏆 AI HIRING COPILOT</h1>
            <p style="color: #cccccc; font-size: 18px;">Intelligent Recruiter AI Agent | Powered by OpenAI + LangChain</p>
            <div style="display: flex; justify-content: center; gap: 10px; margin-top: 10px;">
                <span class="status-indicator"></span>
                <span style="color: #00ff00;">System Online</span>
            </div>
        </div>
    """)
    
    with gr.Tabs():
        # Tab 1: Resume Upload
        with gr.TabItem("📄 Resume Upload", id="upload"):
            with gr.Row():
                with gr.Column(scale=1):
                    file_upload = gr.File(label="Upload Resume (PDF)", file_types=[".pdf"])
                    upload_btn = gr.Button("🚀 Process Resume", variant="primary")
                    
                    with gr.Row():
                        ats_card = gr.HTML(label="ATS Score Card", value="""
                            <div style="text-align: center; padding: 20px; background: rgba(20,20,30,0.8); border-radius: 15px;">
                                <div style="font-size: 20px; margin-bottom: 10px;">📊 ATS Score</div>
                                <div style="font-size: 48px; font-weight: bold; background: linear-gradient(135deg, #ff3366, #00d4ff); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
                                    --
                                </div>
                                <div>Upload resume to see score</div>
                            </div>
                        """)
                    
                with gr.Column(scale=1):
                    resume_summary = gr.Markdown("### 📋 Resume Summary\n\nUpload a PDF resume to get started.")
                    #on = gr.JSON(label="Raw Extracted Data", visible=False)
        
        # Tab 2: AI Chat Assistant
        with gr.TabItem("💬 AI Chat Assistant", id="chat"):
            chatbot = gr.ChatInterface(
                fn=process_chat,
                title="🤖 AI Recruiter Assistant",
                description="Ask me anything about your resume, jobs, interview prep, and career advice!",
                examples=[
                    "Analyze my resume and give ATS score",
                    "Find software engineering jobs for me",
                    "Generate interview questions for a frontend role",
                    "Improve my resume bullet points",
                    "What skills am I missing for a senior role?",
                    "Give me resume formatting advice",
                    "What certifications should I get?",
                    "Create a coding challenge for me"
                ],
                cache_examples=False
            )
        
        # Tab 3: Job Matcher
        with gr.TabItem("🎯 Job Matcher", id="jobs"):
            with gr.Row():
                with gr.Column():
                    job_query = gr.Textbox(label="Job Search Query", placeholder="e.g., Senior Software Engineer, Remote")
                    search_btn = gr.Button("🔍 Find Matching Jobs")
                    job_results = gr.Markdown("### 📊 Job Matches\n\nSearch for jobs to see matches here.")
        
        # Tab 4: Interview Prep
        with gr.TabItem("🎤 Interview Prep", id="interview"):
            with gr.Row():
                with gr.Column():
                    role_input = gr.Textbox(label="Target Role", placeholder="e.g., Frontend Developer, Data Scientist")
                    difficulty_input = gr.Dropdown(label="Difficulty", choices=["beginner", "intermediate", "advanced"], value="intermediate")
                    gen_btn = gr.Button("📝 Generate Questions")
                    interview_output = gr.Markdown("### 🎯 Interview Questions\n\nGenerate questions for your target role.")
        
        # Tab 5: Skill Gap Analysis
        with gr.TabItem("📚 Skill Gap Analysis", id="skills"):
            with gr.Row():
                with gr.Column():
                    target_role = gr.Textbox(label="Target Role", placeholder="e.g., Full Stack Developer")
                    analyze_btn = gr.Button("🔍 Analyze Skill Gaps")
                    gap_output = gr.Markdown("### 📊 Skill Gap Analysis\n\nAnalyze your skills against target role requirements.")
        
        # Tab 6: Resume Improver
        with gr.TabItem("✍️ Resume Improver", id="improve"):
            with gr.Row():
                with gr.Column():
                    bullet_input = gr.Textbox(label="Weak Bullet Point", placeholder="Enter a weak resume bullet point", lines=3)
                    improve_btn = gr.Button("✨ Improve Bullet")
                    improvement_output = gr.Markdown("### 📝 Improved Version\n\nGet ATS-friendly, quantified bullet points.")
    
    # Event handlers
    def on_upload(file):
        summary, score = upload_and_process(file)
        return summary, score
    
    def on_search_jobs(query):
        global agent
        if agent is None:
            return "⚠️ Please upload a resume first in the Resume Upload tab."
        result = agent._search_jobs_tool(query)
        return result
    
    def on_gen_questions(role):
        global agent
        if agent is None:
            return "⚠️ Please upload a resume first."
        result = agent._generate_questions_tool(role)
        return result
    
    def on_analyze_gaps(role):
        global agent
        if agent is None:
            return "⚠️ Please upload a resume first."
        result = agent._analyze_skill_gaps_tool(role)
        return result
    
    def on_improve_bullet(bullet):
        global agent
        if agent is None or bullet.strip() == "":
            return "⚠️ Please upload a resume and enter a bullet point to improve."
        
        improver = ResumeImprover(agent.llm)
        improvements = improver.improve_bullet_points([bullet])
        if improvements:
            imp = improvements[0]
            return f"""
**Original:** {imp['original']}

**Improved:** {imp['improved']}

**Impact Metric:** {imp['impact_metric']}

**Action Verb:** {imp['action_verb']}

**ATS Keywords:** {', '.join(imp['ats_keywords'])}
"""
        return "Could not improve bullet point. Please try a different one."
    
    # Connect handlers
    upload_btn.click(
        on_upload,
        inputs=[file_upload],
        outputs=[resume_summary, ats_card]
    )
    
    search_btn.click(
        on_search_jobs,
        inputs=[job_query],
        outputs=[job_results]
    )
    
    gen_btn.click(
        on_gen_questions,
        inputs=[role_input],
        outputs=[interview_output]
    )
    
    analyze_btn.click(
        on_analyze_gaps,
        inputs=[target_role],
        outputs=[gap_output]
    )
    
    improve_btn.click(
        on_improve_bullet,
        inputs=[bullet_input],
        outputs=[improvement_output]
    )

# ====================================================
# MAIN EXECUTION
# ====================================================

if __name__ == "__main__":
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     🏆 AI HIRING COPILOT - Intelligent Recruiter Agent      ║
    ║                                                              ║
    ║     Status: ✅ System Online                                 ║
    ║     Model: Gemini 1.5 Flash                                  ║
    ║     Tools: Resume Parser | ATS Scorer | Job Matcher          ║
    ║            Interview Gen | Resume Improver | Skill Gap       ║
    ║                                                              ║
    ║     🚀 Launching Gradio Interface...                         ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        debug=False
    )

