import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        "primary": "#81ecff",
        "primary-dim": "#00d4ec",
        "primary-container": "#00e3fd",
        "secondary": "#feb700",
        "secondary-dim": "#ecaa00",
        "tertiary": "#70aaff",
        "surface": "#0e0e0e",
        "surface-container": "#1a1a1a",
        "surface-container-low": "#131313",
        "surface-container-high": "#20201f",
        "on-surface": "#ffffff",
        "on-surface-variant": "#adaaaa",
        "outline": "#767575",
        "outline-variant": "#484847",
      },
      borderRadius: {
        "DEFAULT": "0.25rem",
        "lg": "0.5rem",
        "xl": "0.75rem",
        "full": "9999px"
      },
      fontFamily: {
        "headline": ["var(--font-noto-serif)"],
        "body": ["var(--font-manrope)"],
        "label": ["var(--font-manrope)"]
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
    require('@tailwindcss/container-queries')
  ],
};

export default config;