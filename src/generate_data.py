import os
import random
import pandas as pd

os.makedirs("data", exist_ok=True)

items_data = [
    ("G01", "React Frontend Dashboard", "Web Development", "Medium", "Build a responsive React dashboard with REST API integration and clean UI components.", "react, javascript, responsive ui, rest api, css"),
    ("G02", "Mobile UI Design for Fintech App", "UI/UX Design", "High", "Design modern mobile UI screens and interactive Figma prototypes for an iOS banking app.", "mobile ui, figma, prototyping, ux design, ios"),
    ("G03", "Python Web Scraper for E-commerce", "Data & Automation", "Low", "Develop a Python script using BeautifulSoup and Scrapy to extract product pricing data.", "python, web scraping, beautifulsoup, scrapy, pandas"),
    ("G04", "Full-Stack E-commerce Website", "Web Development", "High", "Create a full-stack online store using React frontend, Node.js backend, and Stripe payments.", "react, nodejs, rest api, mongodb, responsive ui"),
    ("G05", "iOS Fitness Tracker UI Redesign", "UI/UX Design", "Medium", "Redesign mobile UI layout and dark mode components for a fitness tracking application.", "mobile ui, ui design, figma, design systems, android"),
    ("G06", "WordPress Corporate Site Setup", "Web Development", "Low", "Customize a responsive WordPress theme with SEO optimization and contact forms.", "wordpress, php, css, seo, responsive ui"),
    ("G07", "NLP Sentiment Analysis Pipeline", "AI & Machine Learning", "High", "Build a text classification model in Python using scikit-learn and TF-IDF for customer reviews.", "python, nlp, scikit-learn, machine learning, tf-idf"),
    ("G08", "Flutter Cross-Platform Delivery App", "Mobile Development", "High", "Develop an Android and iOS delivery app in Flutter with Firebase authentication and maps.", "flutter, dart, mobile development, firebase, rest api"),
    ("G09", "Power BI Sales Analytics Dashboard", "Data & Automation", "Medium", "Connect SQL Server data to Power BI and create executive sales KPI dashboards.", "power bi, sql, data visualization, dax, analytics"),
    ("G10", "React Native Social App UI", "Mobile Development", "Medium", "Implement responsive mobile UI screens in React Native connected to a GraphQL API.", "react native, javascript, mobile ui, graphql, rest api"),
    ("G11", "Logo and Brand Identity Kit", "Graphic Design", "Low", "Create a minimalist logo, color palette, and typography guide for a coffee startup.", "logo design, branding, illustrator, graphic design"),
    ("G12", "FastAPI Microservice Backend", "Web Development", "Medium", "Build high-performance REST API endpoints using Python FastAPI and PostgreSQL.", "python, fastapi, rest api, postgresql, docker"),
    ("G13", "Mobile E-learning App UX Audit", "UI/UX Design", "Medium", "Conduct UX research and redesign mobile UI onboarding flows for an edtech app.", "mobile ui, ux research, figma, wireframing, prototyping"),
    ("G14", "SQL Database Optimization", "Data & Automation", "Medium", "Optimize slow SQL queries, design indexes, and normalize relational database tables.", "sql, postgresql, database design, performance tuning"),
    ("G15", "Landing Page with Tailwind CSS", "Web Development", "Low", "Convert Figma design into a pixel-perfect responsive landing page using React and Tailwind.", "react, tailwind css, responsive ui, html, css"),
    ("G16", "Computer Vision Defect Detection", "AI & Machine Learning", "High", "Train a PyTorch object detection model to identify manufacturing defects from images.", "python, pytorch, computer vision, deep learning, opencv"),
    ("G17", "SEO Content Writing for SaaS Blog", "Content & Marketing", "Low", "Write 10 technical SEO blog articles about cloud computing and DevOps best practices.", "seo, technical writing, content marketing, copywriting"),
    ("G18", "Android Kotlin Expense Tracker", "Mobile Development", "Medium", "Build an offline-first Android app in Kotlin with Room database and Material UI.", "kotlin, android, mobile development, sqlite, mobile ui"),
    ("G19", "Vue.js Admin Portal Integration", "Web Development", "Medium", "Develop an admin panel in Vue.js consuming external REST API services.", "vuejs, javascript, rest api, responsive ui, css"),
    ("G20", "Figma Design System for Mobile", "UI/UX Design", "High", "Build a reusable component library and mobile UI kit in Figma for a travel platform.", "figma, mobile ui, design systems, ui design, prototyping"),
    ("G21", "Automated Excel Financial Model", "Data & Automation", "Low", "Create automated VBA macros and dynamic financial forecasting templates in Excel.", "excel, vba, financial modeling, data analysis"),
    ("G22", "Django SaaS Subscription Platform", "Web Development", "High", "Develop a multi-tenant SaaS backend in Django with role-based access and REST API.", "python, django, rest api, postgresql, html"),
    ("G23", "Chatbot Integration with OpenAI API", "AI & Machine Learning", "Medium", "Build a customer support chatbot using Python, LangChain, and vector embeddings.", "python, nlp, llm, rest api, chatbot"),
    ("G24", "YouTube Video Editing for Tech Channel", "Video & Media", "Low", "Edit 15-minute tech review videos with motion graphics and color grading.", "video editing, premiere pro, after effects, motion graphics"),
    ("G25", "React TypeScript Component Library", "Web Development", "High", "Architect a strongly-typed React UI library with Storybook documentation and unit tests.", "react, typescript, responsive ui, jest, css"),
    ("G26", "Tableau Customer Churn Dashboard", "Data & Automation", "Medium", "Analyze telecom customer churn data and build interactive Tableau visualizations.", "tableau, sql, data visualization, python, analytics"),
    ("G27", "Social Media Ad Campaign Graphics", "Graphic Design", "Low", "Design 20 high-converting Instagram and Facebook ad banners for a fashion brand.", "graphic design, photoshop, canva, social media, branding"),
    ("G28", "AWS Cloud Infrastructure Setup", "DevOps & Cloud", "High", "Configure AWS EC2, RDS, S3, and CI/CD pipelines using Terraform and Docker.", "aws, terraform, docker, ci/cd, linux"),
    ("G29", "Mobile Food Delivery UI/UX Design", "UI/UX Design", "Medium", "Design intuitive mobile UI checkout and live tracking screens in Figma.", "mobile ui, figma, ui design, ux design, prototyping"),
    ("G30", "Python Recommendation System MVP", "AI & Machine Learning", "High", "Build a hybrid content-based and collaborative filtering recommender in Python.", "python, machine learning, nlp, scikit-learn, pandas"),
    ("G31", "Next.js Portfolio & Blog Platform", "Web Development", "Medium", "Create a fast SSR website with Next.js, React, Markdown support, and REST API.", "nextjs, react, responsive ui, rest api, typescript"),
    ("G32", "iOS SwiftUI Weather Application", "Mobile Development", "Medium", "Develop a sleek iOS app in SwiftUI integrating real-time weather REST APIs.", "swift, swiftui, ios, mobile development, rest api"),
    ("G33", "Google Ads PPC Campaign Audit", "Content & Marketing", "Low", "Audit existing Google Ads account, optimize keywords, and lower cost per click.", "google ads, ppc, digital marketing, analytics, seo"),
    ("G34", "Kubernetes Cluster Migration", "DevOps & Cloud", "High", "Containerize legacy microservices and deploy to managed Kubernetes cluster.", "kubernetes, docker, devops, linux, aws"),
    ("G35", "Product Packaging & Label Design", "Graphic Design", "Medium", "Design print-ready packaging and labels for an organic skincare product line.", "packaging design, illustrator, graphic design, branding"),
    ("G36", "ETL Data Pipeline with Apache Airflow", "Data & Automation", "High", "Build automated daily ETL workflows ingesting API data into BigQuery.", "python, airflow, sql, etl, data engineering"),
    ("G37", "Angular Enterprise CRM Frontend", "Web Development", "High", "Implement complex data tables and responsive UI forms in Angular with REST API.", "angular, typescript, responsive ui, rest api, css"),
    ("G38", "Mobile Banking App Usability Testing", "UI/UX Design", "Low", "Plan and execute remote usability tests for a mobile UI prototype with 10 users.", "ux research, mobile ui, usability testing, figma"),
    ("G39", "C# .NET Desktop Inventory System", "Web Development", "Medium", "Build a Windows WPF desktop application in C# connected to SQL Server.", "c#, .net, wpf, sql, desktop development"),
    ("G40", "Podcast Audio Mixing and Mastering", "Video & Media", "Low", "Clean background noise, balance vocal levels, and master weekly podcast episodes.", "audio editing, adobe audition, podcast, sound design"),
]

items_df = pd.DataFrame(items_data, columns=[
    "item_id", "gig_title", "category", "budget_tier", "project_brief", "required_skills"
])
items_df.to_csv("data/items.csv", index=False)

# 14 freelancers with distinct preferences (each rates 5-6 items -> ~85% sparse, realistic)
# Note: U01 loves Mobile UI Design (5/5) for Test Case 2
ratings_data = [
    ("U01", "G02", 5, "Loved designing fintech mobile UI screens."),
    ("U01", "G05", 5, "Great mobile UI redesign project with clear specs."),
    ("U01", "G13", 5, "Enjoyed the mobile UX audit and wireframing."),
    ("U01", "G01", 3, "Frontend coding was okay but I prefer UI design."),
    ("U01", "G03", 2, "Not interested in backend scraping."),

    ("U02", "G01", 5, "Awesome React and REST API dashboard gig."),
    ("U02", "G04", 5, "Solid full-stack React project."),
    ("U02", "G15", 4, "Quick Tailwind landing page build."),
    ("U02", "G25", 5, "Loved building the React TypeScript library."),
    ("U02", "G11", 2, "Not a graphic designer."),

    ("U03", "G07", 5, "Great NLP sentiment analysis challenge."),
    ("U03", "G16", 5, "Fun PyTorch computer vision task."),
    ("U03", "G23", 4, "Interesting LLM chatbot integration."),
    ("U03", "G30", 5, "Exactly my specialty in recommender systems."),
    ("U03", "G17", 2, "Prefer coding over blog writing."),

    ("U04", "G03", 5, "Clean Python scraping requirements."),
    ("U04", "G09", 4, "Good Power BI dashboard project."),
    ("U04", "G14", 5, "Enjoyed optimizing SQL queries."),
    ("U04", "G26", 4, "Nice churn analytics dataset."),
    ("U04", "G36", 5, "Great Airflow ETL pipeline work."),

    ("U05", "G08", 5, "Smooth Flutter mobile app build."),
    ("U05", "G10", 5, "React Native UI and GraphQL worked well."),
    ("U05", "G18", 4, "Good Kotlin Android project."),
    ("U05", "G32", 4, "Nice SwiftUI weather app."),
    ("U05", "G21", 2, "Excel VBA is not my focus."),

    ("U06", "G11", 5, "Fun startup logo and branding kit."),
    ("U06", "G27", 5, "Creative social media ad graphics."),
    ("U06", "G35", 5, "Loved the skincare packaging design."),
    ("U06", "G02", 4, "Good visual design overlap."),
    ("U06", "G28", 1, "No experience with AWS DevOps."),

    ("U07", "G12", 5, "FastAPI and PostgreSQL is my favorite stack."),
    ("U07", "G22", 5, "Well-scoped Django SaaS backend."),
    ("U07", "G28", 4, "Good AWS infrastructure setup."),
    ("U07", "G34", 5, "Challenging Kubernetes migration."),
    ("U07", "G06", 3, "WordPress is a bit too basic."),

    ("U08", "G01", 4, "Nice React dashboard."),
    ("U08", "G19", 5, "Vue.js admin portal was very clear."),
    ("U08", "G31", 5, "Next.js SSR blog was fun to build."),
    ("U08", "G37", 4, "Solid Angular enterprise frontend."),
    ("U08", "G24", 2, "Not a video editor."),

    ("U09", "G20", 5, "Loved creating the Figma mobile design system."),
    ("U09", "G29", 5, "Food delivery mobile UI was super engaging."),
    ("U09", "G38", 4, "Useful mobile usability testing."),
    ("U09", "G02", 5, "Fintech mobile UI is right up my alley."),
    ("U09", "G14", 2, "Database tuning is outside my skills."),

    ("U10", "G17", 5, "Enjoyed writing SaaS technical SEO blogs."),
    ("U10", "G33", 5, "Great PPC audit with measurable ROI."),
    ("U10", "G06", 4, "Easy WordPress SEO setup."),
    ("U10", "G27", 4, "Good ad creative collaboration."),
    ("U10", "G16", 1, "No deep learning background."),

    ("U11", "G24", 5, "Fun tech YouTube video editing."),
    ("U11", "G40", 5, "Clean podcast audio mixing job."),
    ("U11", "G27", 4, "Enjoyed making social banners."),
    ("U11", "G11", 3, "Decent branding work."),
    ("U11", "G12", 2, "Not a backend developer."),

    ("U12", "G09", 5, "Great Power BI KPIs."),
    ("U12", "G21", 5, "Smooth Excel VBA financial model."),
    ("U12", "G26", 5, "Loved building the Tableau dashboard."),
    ("U12", "G14", 4, "Good SQL optimization task."),
    ("U12", "G08", 2, "Don't build mobile apps."),

    ("U13", "G04", 4, "Good full-stack e-commerce scope."),
    ("U13", "G12", 5, "Clean REST API requirements."),
    ("U13", "G31", 4, "Nice Next.js project."),
    ("U13", "G39", 5, "Solid C# WPF desktop app."),
    ("U13", "G35", 2, "Not a packaging designer."),

    ("U14", "G07", 5, "Strong NLP task with scikit-learn."),
    ("U14", "G23", 5, "Awesome LLM chatbot project."),
    ("U14", "G30", 5, "Loved implementing hybrid recommendations."),
    ("U14", "G36", 4, "Good data engineering pipeline."),
    ("U14", "G40", 1, "Not an audio engineer."),
]

ratings_df = pd.DataFrame(ratings_data, columns=[
    "user_id", "item_id", "rating", "freelancer_note"
])
ratings_df.to_csv("data/ratings.csv", index=False)
print(f"Generated {len(items_df)} items and {len(ratings_df)} ratings.")