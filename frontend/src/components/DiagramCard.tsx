import type { Diagram } from "../types";
import { ExternalLink } from "lucide-react";

interface DiagramCardProps {
  diagram: Diagram;
}

export function DiagramCard({ diagram }: DiagramCardProps) {
  // Convert diagram ID to image path
  // diagram_009 -> /benchmark/corpus/images/diagrams/diagram_009.png
  const imagePath = `/benchmark/corpus/images/diagrams/${diagram.id}.png`;

  return (
    <figure className="my-4 overflow-hidden rounded-xl border border-border bg-white shadow-sm">
      <div className="relative">
        <img
          src={imagePath}
          alt={diagram.vision_description || diagram.description}
          className="w-full object-contain"
          loading="lazy"
        />
      </div>
      <figcaption className="border-t border-border bg-muted/30 px-4 py-3">
        <p className="text-sm text-muted-foreground">
          {diagram.vision_description || diagram.description}
        </p>
        {diagram.post_url && (
          <a
            href={diagram.post_url}
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 inline-flex items-center gap-1 text-xs text-primary hover:underline"
          >
            Source: {diagram.post_title || "View original"}
            <ExternalLink className="h-3 w-3" />
          </a>
        )}
      </figcaption>
    </figure>
  );
}
