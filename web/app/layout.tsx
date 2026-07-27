import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Analytics } from "@vercel/analytics/next";
import "./globals.css";
import { Nav } from "@/components/Nav";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "fastdiff — SDXL inference, honestly measured",
  description:
    "A production-inference image studio: pick a precision × style variant, generate on a real GPU, and watch latency, throughput and VRAM change. Compare quantised variants on one seed.",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="min-h-full flex flex-col">
        <div className="app-backdrop" />
        <Nav />
        <main className="flex-1">{children}</main>
        <Analytics />
      </body>
    </html>
  );
}
