import React from 'react';
import InfoTip from './InfoTip';

const NewsFeed = ({ news, onRead, totalCount }) => {
    const count = totalCount ?? news?.length ?? 0;

    if (!news || news.length === 0) {
        return (
            <div className="h-full flex flex-col bg-transparent p-4">
                <h3 className="text-gold/80 text-[10px] font-bold mb-3 uppercase tracking-widest border-b border-white/10 pb-2">
                    Top News
                </h3>
                <p className="text-slate-500 text-sm flex-1 flex items-center justify-center text-center px-4">
                    {count === 0 ? 'No recent news available for this ticker.' : 'No headlines match your filter.'}
                </p>
            </div>
        );
    }

    return (
        <div className="h-full flex flex-col bg-transparent overflow-hidden">
            <h3 className="text-gold text-[10px] font-bold px-4 pt-3 pb-2 uppercase tracking-widest border-b border-white/10 flex justify-between items-center shrink-0">
                <span className="flex items-center gap-2">
                    Top News
                    <InfoTip dark align="left" title="News Feed">
                        Fresh headlines about this company. Click Read to open in-app, or the title to visit the source.
                    </InfoTip>
                </span>
                <span className="text-slate-500 font-mono">{news.length}{count !== news.length ? ` / ${count}` : ''}</span>
            </h3>
            <ul className="flex-1 overflow-y-auto custom-scrollbar px-2 py-2 space-y-1">
                {news.map((item, index) => (
                    <li
                        key={`${item.link}-${index}`}
                        className="group rounded-lg border border-transparent hover:border-white/10 hover:bg-white/5 p-2.5 transition-colors"
                    >
                        <div className="flex justify-between items-start gap-2 mb-1">
                            <span className="text-[10px] text-gold/70 font-medium truncate">{item.publisher}</span>
                            <span className="text-[10px] text-slate-500 whitespace-nowrap shrink-0">{item.date}</span>
                        </div>
                        <a
                            href={item.link}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="text-xs text-slate-200 group-hover:text-cream font-medium leading-snug line-clamp-3 block mb-2"
                        >
                            {item.title}
                        </a>
                        <button
                            type="button"
                            onClick={() => onRead(item)}
                            className="text-[10px] font-bold uppercase tracking-wider bg-white/5 hover:bg-gold/20 text-slate-300 hover:text-gold px-2 py-1 rounded border border-white/10 transition-colors"
                        >
                            Read
                        </button>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default NewsFeed;
