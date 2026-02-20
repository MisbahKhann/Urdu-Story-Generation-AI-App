import "./globals.css";

export const metadata = {
  title: "داستان گو — Urdu Story Generator",
  description: "Generate Urdu children's stories with AI",
};

export default function RootLayout({ children }) {
  return (
    <html lang="ur" dir="rtl">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        {/* Noto Nastaliq Urdu for authentic Urdu script */}
        {/* Playfair Display for Latin headings */}
        <link
          href="https://fonts.googleapis.com/css2?family=Noto+Nastaliq+Urdu:wght@400;600;700&family=Playfair+Display:ital,wght@0,700;1,400&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}