# Frontend - Next.js Dashboard

This is the frontend dashboard for the Power Load Forecasting project, built with Next.js and Tailwind CSS. It provides an interactive UI for model predictions, health monitoring, and data visualization.

## Features
- Real-time model predictions with probability charts
- Model switching (LightGBM/XGBoost)
- Prediction history with line charts
- Responsive design with dark theme
- Integrates with FastAPI backend for inference

## Prerequisites
```bash
npm install next react react-dom recharts lucide-react
```

## Getting Started
1. Ensure the backend is running at `http://localhost:8080` (or update `BASE_URL` in `app/page.tsx`).
2. Run the development server:
```bash
npm run dev
```
3. Open [http://localhost:3000](http://localhost:3000) to view the dashboard.
 
 Note: if running `backend` locally as well make sure to change the `BASE_URL` in `page.tsx`

## Building for Production
```bash
npm run build
npm start
```

## Project Structure
- `app/page.tsx`: Main dashboard component with forms, charts, and API calls.
- `app/layout.tsx`: Root layout with metadata and fonts.
- `app/globals.css`: Tailwind CSS imports and custom styles.

## Deployment
Deploy to Vercel or similar platforms. Update CORS origins in the backend for production URLs.

## Learn More
- [Next.js Documentation](https://nextjs.org/docs)
- [Recharts for Data Visualization](https://recharts.org/)
- [Tailwind CSS](https://tailwindcss.com/)
