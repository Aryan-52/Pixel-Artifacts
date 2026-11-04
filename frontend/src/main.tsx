import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { createBrowserRouter, Link, Navigate, RouterProvider } from "react-router-dom";
import Layout from "./components/Layout";
import Landing from "./pages/Landing";
import Detect from "./pages/Detect";
import Batch from "./pages/Batch";
import History from "./pages/History";
import Guide from "./pages/Guide";
import Team from "./pages/Team";
import ResearchLayout from "./pages/research/ResearchLayout";
import Methodology from "./pages/research/Methodology";
import Results from "./pages/research/Results";
import Literature from "./pages/research/Literature";
import Publications from "./pages/research/Publications";
import "./styles/global.css";
import "./styles/app.css";

function NotFound() {
  return (
    <section className="wrap page-head">
      <span className="eyebrow">404</span>
      <h1>This page is not here.</h1>
      <p className="lede">The link may be old, or the batch may have expired.</p>
      <p style={{ marginTop: 24 }}><Link className="btn" to="/">Go home</Link></p>
    </section>
  );
}

const router = createBrowserRouter([
  {
    element: <Layout />,
    children: [
      { path: "/", element: <Landing /> },
      { path: "/detect", element: <Detect /> },
      { path: "/batch/:id", element: <Batch /> },
      { path: "/history", element: <History /> },
      { path: "/guide", element: <Guide /> },
      { path: "/team", element: <Team /> },
      {
        path: "/research",
        element: <ResearchLayout />,
        children: [
          { index: true, element: <Navigate to="methodology" replace /> },
          { path: "methodology", element: <Methodology /> },
          { path: "results", element: <Results /> },
          { path: "literature", element: <Literature /> },
          { path: "publications", element: <Publications /> },
        ],
      },
      { path: "*", element: <NotFound /> },
    ],
  },
]);

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
