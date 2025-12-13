import React from 'react';

const NewsFeed = ({ news, onRead }) => {
    if (!news || news.length === 0) {
        return (
            <div className="bg-gray-800 rounded-lg p-4 shadow-lg h-full">
                <h3 className="text-gray-400 text-sm font-semibold mb-3 uppercase tracking-wider border-b border-gray-700 pb-2">News</h3>
                <p className="text-gray-500 text-sm">No recent news available.</p>
            </div>
        );
    }

    return (
        <div className="bg-gray-800 rounded-lg p-4 shadow-lg h-full overflow-y-auto custom-scrollbar">
            <h3 className="text-orange-400 text-sm font-semibold mb-3 uppercase tracking-wider border-b border-gray-700 pb-2 flex justify-between items-center">
                <span>Top News</span>
                <span className="text-xs text-gray-500">{news.length} items</span>
            </h3>
            <ul className="space-y-3">
                {news.map((item, index) => (
                    <li key={index} className="group cursor-pointer hover:bg-gray-750 p-2 -mx-2 rounded transition-colors duration-200">
                        <a href={item.link} target="_blank" rel="noopener noreferrer" className="block">
                            <div className="flex justify-between items-start mb-1">
                                <span className="text-xs text-blue-400 font-medium truncate max-w-[120px]">{item.publisher}</span>
                                <span className="text-xs text-gray-500 whitespace-nowrap">{item.date}</span>
                            </div>
                            <h4 className="text-sm text-gray-200 group-hover:text-blue-300 font-medium leading-tight mb-1">
                                {item.title}
                            </h4>

                            <div className="flex gap-2 mt-1">
                                <button
                                    onClick={(e) => {
                                        e.preventDefault();
                                        e.stopPropagation();
                                        onRead(item);
                                    }}
                                    className="text-xs bg-gray-700 hover:bg-gray-600 text-white px-2 py-1 rounded transition-colors"
                                >
                                    Read Article
                                </button>
                            </div>
                        </a>
                    </li>
                ))}
            </ul>
        </div>
    );
};

export default NewsFeed;
