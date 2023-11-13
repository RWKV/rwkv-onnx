#ifndef BYTEMATCH_HPP
#define BYTEMATCH_HPP

#include <string>
#include <vector>
#include <unordered_map>
#include <set>
#include <iostream>
#include <fstream>
#include <sstream>
#include "bytematch.hpp"


class RWKVTokenizer {
private:
    std::vector<std::vector<std::vector<std::vector<uint8>>>> table;
    std::vector<std::set<int>> good;
    std::vector<int> wlen;
    std::unordered_map<int, std::vector<uint8>> idx2token;
    std::unordered_map<std::string, int> token2idx;
    

public:
    RWKVTokenizer(const std::string& fileName) {
        // Reading file and constructing idx2token and token2idx
        std::ifstream file(fileName);
        std::string line;
        std::vector<std::vector<uint8>> sorted;
        while (std::getline(file, line)) {
            size_t idxSpace = line.find(' ');
            int idx = std::stoi(line.substr(0, idxSpace));
            auto x = findPythonByteObjects(line.substr(idxSpace + 1, line.rfind(' ') - idxSpace - 1));
            if (x.size() == 0) {
                // decode the string as utf-8
                auto line2 = line.substr(idxSpace + 2, line.rfind(' ') - idxSpace - 3);
                // std::cout << line2 << std::endl;
                x = std::vector<uint8>(line2.begin(), line2.end());
            }
            
            sorted.push_back(x);
            idx2token[idx] = x;
        }

        file.close();

        // Constructing token2idx
        for (const auto& pair : idx2token) {
            std::string tokenStr(pair.second.begin(), pair.second.end());
            token2idx[tokenStr] = pair.first;
        }

        // precompute some tables for fast matching
        // this.table = Array.from({ length: 256 }, () => Array.from({ length: 256 }, () => []));
        // this.good = Array.from({ length: 256 }, () => new Set<number>());
        // this.wlen = Array.from({ length: 256 }, () => 0);

        // for (let i = sorted.length - 1; i >= 0; i--) { // reverse order - match longer tokens first
        //     const s = sorted[i];
        //     if (s.length >= 2) {
        //         const s0 = s[0];
        //         const s1 = s[1];
        //         this.table[s0][s1].push(s);
        //         this.wlen[s0] = Math.max(this.wlen[s0], s.length);
        //         this.good[s0].add(s1);
        //     }
        // } Convert to c++
        table = std::vector<std::vector<std::vector<std::vector<uint8>>>>(256, std::vector<std::vector<std::vector<uint8>>>(256, std::vector<std::vector<uint8>>()));
        good = std::vector<std::set<int>>(256, std::set<int>());
        wlen = std::vector<int>(256, 0);

        for (int i = sorted.size() - 1; i >= 0; i--) {
            const std::vector<uint8>& s = sorted[i];
            if (s.size() >= 2) {
                const uint8 s0 = s[0];
                const uint8 s1 = s[1];
                // init table[s0][s1] if it doesn't exist
                
                table[s0][s1].push_back(s);
                wlen[s0] = std::max(wlen[s0], (int)s.size());
                good[s0].insert(s1);
            }
        }

        // More initializer code would go here to replicate the JavaScript behavior
    }

    // More methods to replicate the JavaScript behavior would go here

    // Sample print function based on printTokens
    void printTokens(const std::vector<int>& tokens) {
        for (auto i : tokens) {
            try {
                std::vector<uint8> s = idx2token[i];
                std::string str(s.begin(), s.end());
                std::cout << "\"" << str << "\"" << i << " ";
            } catch (...) {
                // If the conversion to string fails, keep it in some other format.
                // Note: Better error handling is needed
                std::cout << "Error" << i << " ";
            }
        }
        std::cout << std::endl;
    }


    std::vector<int> encode(const std::string &src) {
        std::vector<uint8> srcBytes(src.begin(), src.end());
        return encodeBytes(srcBytes);
    }

    std::string decode(const std::vector<int> &tokens) {
        std::vector<uint8> byteResult = decodeBytes(tokens);
        return std::string(byteResult.begin(), byteResult.end());
    }

    static bool startsWith(const std::vector<uint8> &target, const std::vector<uint8> &prefix) {
        if (prefix.size() > target.size()) {
            return false;
        }
        return std::equal(prefix.begin(), prefix.end(), target.begin());
    }

private:
    std::vector<int> encodeBytes(const std::vector<uint8>& src) {
        const size_t srcLen = src.size();
        std::vector<int> tokens;
        size_t i = 0;
        while (i < srcLen) {
            std::vector<uint8> s(src.begin() + i, src.begin() + i + 1);
            if (i < srcLen - 1) {
                const int s1 = src[i + 1];
                const int s0 = src[i];
                if (good[s0].find(s1) != good[s0].end()) {
                    std::vector<uint8> sss(src.begin() + i, src.begin() + i + wlen[s0]);
                    auto matchIt = std::find_if(table[s0][s1].begin(), table[s0][s1].end(),
                        [&sss](const std::vector<uint8>& t) { return startsWith(sss, t); });
                    if (matchIt != table[s0][s1].end()) {
                        s = *matchIt;
                    }
                }
            }
            std::string sStr(s.begin(), s.end());
            tokens.push_back(token2idx[sStr]);
            i += s.size();
        }
        return tokens;
    }

    std::vector<uint8> decodeBytes(const std::vector<int> &tokens) {
        std::vector<uint8> decoded;
        for (int token : tokens) {
            const std::vector<uint8> &tokenBytes = idx2token.at(token);
            decoded.insert(decoded.end(), tokenBytes.begin(), tokenBytes.end());
        }
        return decoded;
    }

    

};

RWKVTokenizer worldTokenizer("rwkv_vocab_v20230424.txt");
// int main() {
    
//     std::vector<int> sampleTokens = worldTokenizer.encode("Hello World!");
//     std::cout << "Encoded tokens: " << sampleTokens[0] << " " << sampleTokens[1] << std::endl;
//     std::cout << worldTokenizer.decode({33155, 37576}) << std::endl;
//     return 0;
// }

#endif // BYTEMATCH_HPP