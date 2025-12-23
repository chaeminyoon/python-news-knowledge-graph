import React, { useState } from 'react';
import './SearchBar.css';

function SearchBar({ onSearch, loading }) {
    const [query, setQuery] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();
        if (query.trim() && !loading) {
            onSearch(query);
        }
    };

    return (
        <form className="search-bar" onSubmit={handleSubmit}>
            <div className="search-input-container">
                <input
                    type="text"
                    className="search-input"
                    placeholder="궁금한 뉴스를 검색해보세요... (예: 경제 분야 최신 소식)"
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    disabled={loading}
                />
                <button
                    type="submit"
                    className="search-button"
                    disabled={loading || !query.trim()}
                >
                    {loading ? '검색 중...' : '🔍 검색'}
                </button>
            </div>
        </form>
    );
}

export default SearchBar;
