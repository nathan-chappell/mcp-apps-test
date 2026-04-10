import { createTheme } from "@mantine/core";

export const appTheme = createTheme({
  primaryColor: "teal",
  fontFamily: '"Avenir Next", "Segoe UI", "Helvetica Neue", sans-serif',
  headings: {
    fontFamily: '"Space Grotesk", "Avenir Next", sans-serif',
  },
  radius: {
    xs: "10px",
    sm: "14px",
    md: "18px",
    lg: "24px",
  },
  shadows: {
    sm: "0 10px 28px rgba(26, 52, 65, 0.08)",
    md: "0 18px 42px rgba(26, 52, 65, 0.12)",
  },
  colors: {
    sand: [
      "#f8f7f1",
      "#f1eddc",
      "#e6dfc5",
      "#d6cba4",
      "#c7b985",
      "#bdad70",
      "#b7a565",
      "#9f8e52",
      "#8e7e47",
      "#796a39",
    ],
    ink: [
      "#eef5f6",
      "#dbe6e8",
      "#b6cfd4",
      "#90b8bf",
      "#6ea3ad",
      "#5a96a1",
      "#4a7f8a",
      "#3b6670",
      "#2d4d56",
      "#1e353b",
    ],
  },
  other: {
    pageGradient:
      "linear-gradient(180deg, rgba(248, 247, 241, 0.94) 0%, rgba(241, 237, 220, 0.82) 52%, rgba(219, 230, 232, 0.88) 100%)",
  },
});
