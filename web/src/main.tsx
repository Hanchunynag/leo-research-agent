import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import ScholarConsole from "./ScholarConsole";
import "./styles.css";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <ScholarConsole />
  </StrictMode>,
);
